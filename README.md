# マルチAIエージェントシステム 🤖

LangChain、LangGraph、Streamlitを使用して構築された、AIエージェントが協調してタスクを解決するマルチエージェントシステムです。

## 概要

このシステムは、ユーザーの指示を理解し、複数の専門AIエージェントにタスクを割り振って実行させます。

  * **Supervisor (マネージャー)**: ユーザーの要求を分析し、タスクの計画を立て、適切なワーカーに指示を出します。
  * **Webサーファー (ワーカー)**: Web検索と情報収集を担当します。
  * **ファイルオペレーター (ワーカー)**: ファイルの読み書きや保存を担当します。

例えば、「ミスタードーナツの期間限定商品を調べてCSVファイルにまとめて」といった、Web検索とファイル操作を組み合わせた複雑なタスクを一度の指示で自動実行します。

### 処理フロー
<img width="1171" height="1159" alt="sequence_diagram" src="https://github.com/user-attachments/assets/ac0775d1-06b7-49c1-a199-d41daaa7e42e" />

---

## 🛠️ 主な使用技術

  * **バックエンド**: LangChain, LangGraph, Google Gemini
  * **フロントエンド**: Streamlit
  * **Agent Tools**: Playwright MCP、Filesystem MCP

---

## 🚀 セットアップと実行方法

### 1\. 前提条件

  * Python 3.9以上
  * Node.jsがインストールされていること（npxが使えること）

### 2\. リポジトリのクローン

```bash
git clone https://github.com/hiratsukaaa682/multi-ai-agent.git
cd multi-ai-agent
```

### 3\. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

> `requirements.txt`の内容はプロジェクトファイルを参照してください。

### 4\. 環境変数の設定

プロジェクトのルートに`.env`ファイルを作成し、Google APIキーを記述します。

```.env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

### 5\. ツールの設定

`mcp_config.json`内のファイルパスを、**あなたの環境の絶対パスに書き換えてください**。これはエージェントがファイルを操作する対象のディレクトリです。

**`mcp_config.json`**

```json
{
    "mcpServers": {
      "web-search": {
        "command": "npx",
        "args": [
          "@playwright/mcp@latest"
        ],
        "transport": "stdio"
      },
      "file-system": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "/path/to/your/project/multi-agent-system/output"  // <-- このパスを必ず変更！
        ],
        "transport": "stdio"
      }
    }
}
```

### 6\. アプリケーションの実行

以下のコマンドでStreamlitアプリを起動します。

```bash
streamlit run multi_ai_agent.py
```

ブラウザで `http://localhost:8501` を開きます。

---

## 📂 プロジェクト構成

```
multi-ai-agent/
├── conversation_history/   # 会話履歴の保存先
├── output/                 # ファイル操作の対象ディレクトリ
├── .env                    # 環境変数ファイル
├── mcp_config.json         # MCPツール設定
├── multi_ai_agent.py       # メインアプリケーション
├── agent_conversation.log  # エージェントの動作ログ
└── requirements.txt        # 依存ライブラリ
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md) file for details.
