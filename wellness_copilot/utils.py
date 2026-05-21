from langgraph.prebuilt import create_react_agent

# --- 辅助函数：创建一个专家 Agent 节点 ---
def create_agent(llm, tools, system_prompt):
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
