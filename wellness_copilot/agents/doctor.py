"""Doctor expert — medical-advice child agent invoked by Orchestrator."""
import os
import re

from langchain_core.messages import HumanMessage

from ..detail import print_expert_end, print_expert_start, print_expert_trace
from ..llm import extract_text_content, llm
from ..mcp_client import MCP_REGISTRY
from ..personalization import (
    build_personalization_ctx,
    build_personalization_decision_points,
    format_decision_points_for_prompt,
)
from ..tools import retrieve_doctor_knowledge
from ..utils import create_agent
from .. import isolation
from ._scratchpad import build_scratchpad_note
from .fallbacks import expert_error_update


DOCTOR_DISCLAIMER = "仅供参考，如有不适请就医。"


_DIAGNOSIS_OR_PRESCRIPTION = re.compile(
    r"是什么病|诊断|确诊|可能是|是不是.*病|帮我诊断|处方|开药|开一张|"
    r"剂量|吃多少药|服药量|抗生素|布洛芬|对乙酰氨基酚|阿司匹林",
    re.IGNORECASE,
)

_MEDICATION_NAME = re.compile(r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|药物相互作用", re.IGNORECASE)

_URGENT_SIGNAL = re.compile(
    r"胸痛|胸闷|呼吸困难|喘不上|晕厥|昏厥|头晕|晕晕|眩晕|恶心|呕吐|"
    r"心悸|心慌|心率异常|心律不齐|血压|血糖|剧痛|持续疼痛|疼了|痛了|"
    r"酸痛|肌肉酸|腿酸|胳膊酸|肩酸|腰酸|肿胀|发热|麻木|无力|"
    r"大小便异常|黑便|过敏反应",
    re.IGNORECASE,
)

_EXERCISE_SYMPTOM_SIGNAL = re.compile(
    r"(?:运动|训练|跑步|健身|锻炼|练腿|练胸|练背|力量|有氧|HIIT|肌肉).{0,30}"
    r"(?:头晕|晕晕|眩晕|恶心|呕吐|胸痛|胸闷|心悸|心慌|酸痛|疼痛|痛|疼|肿胀|无力)"
    r"|(?:头晕|晕晕|眩晕|恶心|呕吐|胸痛|胸闷|心悸|心慌|酸痛|疼痛|痛|疼|肿胀|无力).{0,30}"
    r"(?:运动|训练|跑步|健身|锻炼|练腿|练胸|练背|力量|有氧|HIIT|肌肉)",
    re.IGNORECASE,
)


def ensure_doctor_disclaimer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return DOCTOR_DISCLAIMER
    if DOCTOR_DISCLAIMER in text or "仅供参考" in text and "就医" in text:
        return text
    return f"{text}\n\n{DOCTOR_DISCLAIMER}"


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _medical_profile_intro(pctx: dict, user_question: str, answer: str) -> str:
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    injuries = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]
    anchors = []
    for key, suffix in (("age", "岁"), ("weight", "kg"), ("height", "cm")):
        try:
            value = float(stats.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            anchors.append(f"{int(value) if value.is_integer() else round(value, 1)}{suffix}")

    add = []
    text = answer or ""
    if anchors and not any(a.rstrip("岁kgcm") in text for a in anchors):
        add.append(f"结合你目前 {', '.join(anchors)}，医学建议需要以面诊评估为准。")
    if injuries and not any(i in text for i in injuries):
        add.append(f"同时要考虑你记录里的 {', '.join(injuries)}，不要自行加重相关部位负荷或按未确认诊断处理。")
    if not add:
        return answer
    return "\n\n".join(add + [answer])


def _deterministic_doctor_answer(pctx: dict, user_question: str) -> str:
    q = user_question or ""
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    age = stats.get("age")
    weight = stats.get("weight")
    height = stats.get("height")
    anchors = []
    for value, suffix in ((age, "岁"), (weight, "kg"), (height, "cm")):
        try:
            n = float(value or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            anchors.append(f"{int(n) if n.is_integer() else round(n, 1)}{suffix}")
    anchor_text = f"结合你目前 {', '.join(anchors)}，" if anchors else ""
    try:
        weight_n = float(weight or 0)
        height_n = float(height or 0)
    except (TypeError, ValueError):
        weight_n = height_n = 0
    bmi = round(weight_n / ((height_n / 100) ** 2), 1) if weight_n > 0 and height_n > 0 else None

    if re.search(r"糖尿病|血糖", q, re.IGNORECASE) and re.search(r"HIIT|高强度|间歇", q, re.IGNORECASE):
        bmi_text = f"，BMI约{bmi}" if bmi else ""
        return ensure_doctor_disclaimer(
            f"{anchor_text}你有糖尿病，又想直接开始30分钟HIIT，这不建议直接做。"
            f"以你目前体重 {weight_n:g}kg{bmi_text} 来看，突然上高强度间歇会同时增加血糖波动、低血糖和心血管负担风险。\n\n"
            "开始前先让医生评估运动许可、血糖控制、用药时机、并发症、足部保护和心血管风险。"
            "运动前后都要监测血糖，并准备可快速处理低血糖的食物。\n\n"
            "更稳的起点是每周3-4次、每次20-30分钟低到中等强度运动，例如快走、游泳或低阻力固定单车；"
            "等血糖反应稳定且医生许可后，再逐步尝试更短的间歇训练，而不是一开始就做30分钟HIIT。"
        )

    if re.search(r"血压.*(?:大重量|深蹲|增肌|力量)|(?:大重量|深蹲|增肌|力量).*血压", q, re.IGNORECASE):
        return ensure_doctor_disclaimer(
            f"{anchor_text}血压控制不稳定时，不建议直接做大重量深蹲增肌。"
            "大重量深蹲常伴随憋气和瞬间血压升高，可能增加心脑血管风险；在血压稳定并获得医生评估前，先避免 1RM、低次数大重量、力竭组和长时间憋气。\n\n"
            "更稳妥的做法是先去心内科/全科确认血压控制目标和运动许可；训练上暂时选择低到中等负荷、可顺畅呼吸的动作，"
            "强度控制在 RPE 5-6，每组保留 3-4 次余力，组间休息 2-3 分钟，并记录训练前后血压。"
            "如果训练中出现胸闷、胸痛、头晕、心悸或血压明显升高，应立即停止并就医。"
        )

    if re.search(r"尿色.*浓茶|浓茶.*尿|茶色尿|酱油色尿|尿.*茶", q, re.IGNORECASE):
        return ensure_doctor_disclaimer(
            f"{anchor_text}高强度训练后全身肌肉明显疼痛并出现浓茶样尿，不应当按普通酸痛处理，"
            "需要警惕横纹肌溶解及肾损伤风险。请现在先停止训练，尽快去急诊或医院评估。\n\n"
            "就诊时建议说明昨天训练内容、持续时间、补水情况和尿色变化；常见需要检查尿常规、肌酸激酶 CK、肌酐/肾功能、电解质等。"
            "在医生排除风险前，不要继续运动、不要饮酒，也不要自行叠加止痛药。"
        )

    if _DIAGNOSIS_OR_PRESCRIPTION.search(q):
        if re.search(r"处方|开药|开一张|剂量|吃多少药|服药量", q):
            return ensure_doctor_disclaimer(
                f"{anchor_text}我不能开处方，也不能替医生决定具体药物或剂量。请尽快找医生、全科门诊或相应专科做面诊，"
                "由医生结合症状、查体、既往病史、用药禁忌和必要检查来判断。\n\n"
                "在获得专业意见前，不要自行加量、延长疗程或把止痛药当作继续训练的许可；"
                "若出现胸痛胸闷、呼吸困难、黑便、明显过敏反应、麻木无力或疼痛快速加重，请及时就医。"
            )
        if _MEDICATION_NAME.search(q):
            return ensure_doctor_disclaimer(
                f"{anchor_text}我不能仅凭这段描述判断布洛芬是否还能继续用，也不能替医生或药师决定剂量、加量或延长疗程。"
                "连续使用 5 天后仍有腰痛，建议尽快咨询医生/药师或骨科、康复科评估疼痛原因和用药安全。\n\n"
                "布洛芬这类药需要考虑胃肠道出血、肾功能、血压/心血管风险、过敏史以及是否和其他药物叠加。"
                "在没有专业意见前，不要自行继续用药、加量、叠加同类止痛药或把止痛药当作继续训练的许可；"
                "若出现黑便、明显胃痛、胸闷、呼吸困难、皮疹/面唇肿胀、下肢麻木无力或大小便异常，请及时就医。"
            )
        return ensure_doctor_disclaimer(
            f"{anchor_text}我不能根据文字判断你“是什么病”，也不能做诊断。持续或反复的疼痛/不适需要医生查体，"
            "必要时结合影像、心电图或实验室检查来明确。\n\n"
            "在明确原因前，先暂停剧烈运动和会诱发症状的动作，不要自行按某个诊断用药。"
            "如果伴随胸痛胸闷、呼吸困难、发热、晕厥、疼痛快速加重或神经症状，请优先急诊。"
        )

    if _EXERCISE_SYMPTOM_SIGNAL.search(q):
        return ensure_doctor_disclaimer(
            f"{anchor_text}你描述的是运动/健身后出现身体不适，尤其伴有头晕，不能只按普通训练酸痛处理。"
            "建议先停止高强度训练和继续加量，今天以休息、补水、正常进食和观察症状为主；"
            "如果头晕持续、反复出现，或伴随胸闷胸痛、呼吸困难、心悸、明显乏力、呕吐、晕厥、肌肉肿胀明显、尿色像浓茶，"
            "请尽快就医或急诊评估。\n\n"
            "就医前可以记录：最近一周训练频率和强度、出汗量、饮水和进食情况、头晕出现时间、持续多久、是否伴随心悸/胸闷/恶心、"
            "以及尿色和肌肉疼痛是否越来越重。"
        )

    if _URGENT_SIGNAL.search(q):
        return ensure_doctor_disclaimer(
            f"{anchor_text}这个描述里有需要谨慎处理的身体症状，建议先做医生评估，不要只按训练疲劳或普通不适处理。"
            "如果症状正在发生、反复出现或影响日常活动，请尽快去全科/急诊/相应专科；"
            "若有胸痛胸闷、呼吸困难、晕厥、明显心率异常、神经症状或疼痛快速加重，应优先急诊。\n\n"
            "等待就医前，先停止高强度训练、饮酒和自行叠加用药，记录症状出现时间、诱因、持续多久、伴随症状和已服药物，方便医生判断。"
        )

    return ""


def _build_doctor_agent(pctx: dict, peer_notes_text: str, episode_context: str = "", user_question: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = (pctx.get("role_user_cards") or {}).get("Doctor") or pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    decision_section = format_decision_points_for_prompt(
        build_personalization_decision_points(pctx, user_question, role="Doctor")
    )
    mcp_tools = MCP_REGISTRY.get_tools("medical")
    medical_tools = [retrieve_doctor_knowledge, *mcp_tools]
    mcp_hint = (
        "如需权威医学参考，可调用 medical MCP 工具。优先使用 search-medical-literature / "
        "search-clinical-guidelines / search-drugs / get-drug-details / search-drug-nomenclature；"
        "检索式尽量用简洁英文关键词。"
        if mcp_tools
        else ""
    )
    system_prompt = (
        "你是医学顾问 Doctor 子 agent，负责提供一般医学信息、症状风险分层、就医建议和就诊前准备建议。\n\n"
        f"{user_card}\n"
        f"{decision_section}"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        f"{isolation.noniso_history_section(pctx)}"
        "用户卡片就是本轮可用画像；不要为了读取画像而调用工具。"
        "如需要本地医学边界、症状分诊、慢病指标或健康筛查语料，可主动调用 retrieve_doctor_knowledge。"
        f"{mcp_hint}"
        "你必须遵守医疗边界：不能诊断疾病，不能开处方，不能给处方药或需个体化评估药物的具体剂量，"
        "不能保证“没事”。当用户询问诊断、处方、药物剂量、持续疼痛、胸痛/胸闷、呼吸困难、晕厥、"
        "明显心率异常、血压/血糖异常、孕哺期用药或过敏反应时，先建议就医或医生/药师评估。"
        "在安全边界之后，可以提供就诊前记录什么、暂时避免什么、哪些红旗需要急诊。"
        "如果使用 medical MCP 的检索结果，请用中文概括，不要堆砌英文文献。"
        "回答结尾必须包含这句话：仅供参考，如有不适请就医。"
    )
    return create_agent(llm, medical_tools, system_prompt)


def run_doctor(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    try:
        os.environ["WELLNESS_COPILOT_USER_ID"] = user_id
        print_expert_start("Doctor", user_question)
        pctx = pctx or build_personalization_ctx(user_id)
        deterministic = _deterministic_doctor_answer(pctx, user_question)
        if deterministic:
            print_expert_end("Doctor", [], deterministic)
            return {
                "expert_responses": {"Doctor": deterministic},
                "agent_notes": {"Doctor": build_scratchpad_note("Doctor", deterministic)},
                "last_tools": [],
                "retrieval_hits": 0,
            }

        agent = _build_doctor_agent(pctx, peer_notes_text, episode_context, user_question)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("Doctor", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        answer = extract_text_content(result["messages"][-1])
        answer = _medical_profile_intro(pctx, user_question, answer)
        answer = ensure_doctor_disclaimer(answer)
        print_expert_end("Doctor", used_tools, answer)
        return {
            "expert_responses": {"Doctor": answer},
            "agent_notes": {"Doctor": build_scratchpad_note("Doctor", answer)},
            "last_tools": used_tools,
            "retrieval_hits": 0,
        }
    except Exception as e:
        update = expert_error_update("Doctor", e)
        answer = ensure_doctor_disclaimer((update.get("expert_responses") or {}).get("Doctor", ""))
        update["expert_responses"] = {"Doctor": answer}
        update["agent_notes"] = {"Doctor": build_scratchpad_note("Doctor", answer)}
        return update
