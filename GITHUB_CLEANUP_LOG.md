# Wellness Copilot - GitHub 推送清理清单

## ✅ 已完成的清理

### 1. Git安全性
- ✅ 从Git缓存移除了 `.env` 文件（包含API密钥）
- ✅ 创建了 `.env.example` 作为模板
- ✅ 创建了完整的 `.gitignore` 文件

### 2. 忽略的文件夹/文件

**.gitignore** 包含以下规则：

| 类别 | 文件/文件夹 | 原因 |
|------|-----------|------|
| **敏感信息** | `.env` | 包含API密钥 |
| | `profile_store.json` | 用户个人数据 |
| | `.claude/` | 本地配置 |
| **缓存** | `hf_cache/` | Hugging Face模型缓存 (~2GB) |
| | `__pycache__/` | Python编译缓存 |
| | `*.db` | 数据库文件 |
| **开发** | `.vscode/` | 编辑器配置 |
| | `venv/`, `env/` | Python虚拟环境 |

## 📋 使用步骤

### 首次设置

1. **克隆仓库**
   ```bash
   git clone <your-repo-url>
   cd wellness-copilot
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env，填入你的API密钥
   ```

3. **创建Python环境**
   ```bash
   conda env create -f environment.yml
   conda activate wellness-copilot-rag
   ```

4. **下载RAG模型**
   ```bash
   python scripts/download_rag_models.py
   ```

5. **运行项目**
   ```bash
   python main.py
   ```

## 🔐 敏感信息检查

推送前验证：

```bash
# 确保没有追踪敏感文件
git ls-files | grep -E "\.env$|profile_store\.json|\.db$|hf_cache"

# 应该没有输出
```

## 📝 相关文件

- **`.env.example`** - 环境变量模板
- **`.gitignore`** - Git忽略规则
- **`README.md`** - 已更新设置说明
