"""ReplanJudge — meta-LLM that decides whether to bring in another expert.

Runs after every expert and before the Dispatcher. Looks at the user's
question, the expert's answer, and who has already been consulted, then
returns a structured verdict:

  VERDICT: CONTINUE
  ---
  VERDICT: REPLAN
  REASON: <single sentence>

The judge is intentionally separate from the experts so the experts can
focus on their answer quality without having to self-assess scope. A
dedicated judge is more reliable than asking each expert to emit a marker
in its own output.

Cap is enforced by Dispatcher, not here — this judge only emits a request.
"""
from langchain_core.messages import HumanMessage, SystemMessage
import re

from ..llm import extract_text_content, llm
from .query_rewriter import get_user_question


_VALID_EXPERTS = {"Trainer", "Nutritionist", "Psychologist", "Doctor"}

# Cap mirrors dispatcher.REPLAN_CAP; once we've spent all replan slots there
# is nothing more the judge can usefully ask for. Avoids a Dispatcher↔Judge
# loop when the judge keeps wanting "one more".
_REPLAN_CAP = 2

_EXPERT_LABELS = {
    "Trainer": "训练教练（运动/动作/恢复）",
    "Nutritionist": "营养师（饮食/热量/营养素）",
    "Psychologist": "心理疗愈师（压力/焦虑/情绪/倦怠/心理安全）",
    "Doctor": "医学顾问（症状风险/就医/用药边界）",
}

_CROSS_DOMAIN_SIGNAL = re.compile(
    r"吃.*练|练.*吃|饮食.*训练|训练.*饮食|运动.*饮食|饮食.*运动|"
    r"睡眠.*饮食|饮食.*睡眠|失眠.*饮食|压力.*吃|吃.*恢复|练.*恢复|"
    r"10\s*(?:K|公里)|5\s*(?:K|公里)|半马|马拉松|补给|碳水加载|"
    r"伤病.*饮食|饮食.*伤病|康复.*饮食|饮食.*康复",
    re.IGNORECASE,
)

_SIMPLE_SINGLE_DOMAIN = {
    "Trainer": re.compile(r"TDEE|BMR|基础代谢|每日所需热量|深蹲|开始做.*蹲|练胸|走多少步|ACL|半月板|肩袖", re.IGNORECASE),
    "Nutritionist": re.compile(r"肌酸|蛋白|高蛋白早餐|食谱|早餐|午餐|晚餐|训练前|训练后|运动前|运动后", re.IGNORECASE),
    "Psychologist": re.compile(r"睡不着|失眠|入睡|没动力|运动的欲望|焦虑|紧张|压力", re.IGNORECASE),
    "Doctor": re.compile(r"胸痛|胸闷|呼吸困难|晕厥|心率|血压|血糖|诊断|是什么病|处方|剂量|用药", re.IGNORECASE),
}


_JUDGE_SYSTEM = """\
你是健康团队的「补员判官」。你的唯一职责是：判断刚刚结束工作的专家所给的回答，是否完整覆盖了用户的需求；如果**确实有跨领域的核心诉求**没人回答，就请求追加一位专家。

判断标准要克制：
- 单一领域的小细节缺失 → 不追加（CONTINUE）
- 用户问题里隐含的、但当前专家不擅长的另一个领域 → 追加（REPLAN）
- 已经派出过的角色无论如何不要再追加
- 同一轮内最多追加 2 个专家（你不用记次数，上游有限制；你只判断本次是否需要补一个）

可选角色及其专长：
- Trainer: 训练/动作/运动恢复
- Nutritionist: 饮食/营养/热量
- Psychologist: 心理疗愈，处理压力/焦虑/情绪/倦怠/压力性进食/心理安全，不处理身体症状
- Doctor: 医学建议/症状风险/就医建议/用药与诊断边界

输出格式（严格遵守）：
- 不需要追加：仅一行 `VERDICT: CONTINUE`
- 需要追加：两行
  ```
  VERDICT: REPLAN
  REASON: <一句话说明为什么需要补叫哪个领域的专家>
  ```
"""


def _format_executed(executed):
    if not executed:
        return "（暂无）"
    return ", ".join(executed)


def _parse_verdict(text: str) -> str:
    """Return the REASON if REPLAN, else empty string."""
    if not text:
        return ""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    head = lines[0].upper().replace(" ", "")
    if not head.startswith("VERDICT:REPLAN"):
        return ""
    # Find REASON line (case-insensitive)
    for line in lines[1:]:
        if line.upper().startswith("REASON:"):
            return line[len("REASON:"):].strip().lstrip(" :：")
    # REPLAN without REASON — synthesize a generic reason so dispatcher still fires.
    return "判官判定需要补叫其他专家，但未给出具体理由"


def replan_judge_node(state):
    # Remaining planned experts already decided by Orchestrator — nothing to replan yet.
    if state.get("plan"):
        return {}

    # Cap reached — any further replan request would be dropped by the
    # Dispatcher and create a routing loop. Skip the LLM call entirely.
    if int(state.get("replan_count", 0) or 0) >= _REPLAN_CAP:
        return {}

    executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
    if not executed:
        # Nothing to judge yet.
        return {}
    if len(executed) >= 2:
        # Multi-specialist plans already carry cross-domain coverage; let
        # Aggregator/Critic handle residual quality gaps instead of spending
        # another meta-LLM call that often over-adds a third expert.
        return {}

    last_expert = executed[-1]
    if last_expert not in _VALID_EXPERTS:
        return {}

    responses = state.get("expert_responses") or {}
    last_answer = responses.get(last_expert, "")
    if not last_answer:
        return {}

    user_question = get_user_question(state) or "（未获取到原始问题）"

    if last_expert == "Nutritionist" and re.search(r"训练前|训练后|运动前|运动后", user_question):
        return {}

    if not _CROSS_DOMAIN_SIGNAL.search(user_question):
        simple_pat = _SIMPLE_SINGLE_DOMAIN.get(last_expert)
        if simple_pat and simple_pat.search(user_question):
            return {}

    review_prompt = (
        f"用户问题：\n{user_question}\n\n"
        f"刚完成工作的专家：{last_expert}（{_EXPERT_LABELS.get(last_expert, last_expert)}）\n"
        f"该专家的回答：\n{last_answer}\n\n"
        f"本轮已经派出过的专家：{_format_executed(executed)}\n"
        f"还有计划但未执行的专家：{state.get('plan') or '（无）'}\n\n"
        "请按指定格式给出判断。"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=review_prompt),
        ])
        raw = extract_text_content(response)
    except Exception:
        # Judge failure must not block the flow — just continue.
        return {}

    reason = _parse_verdict(raw)
    if not reason:
        return {}

    return {"replan_request": reason}
