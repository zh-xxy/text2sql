#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const PROJECT_ROOT = process.argv[2]
  ? path.resolve(process.argv[2])
  : process.cwd();

const CODE_OVERRIDES = {
  "__init__.py": {
    fileSummary: "text2sql Python 包的标记文件。",
    tags: ["python", "package"],
    complexity: "simple",
  },
  "chatbot.py": {
    fileSummary:
      "会话入口，负责缓存 schema 文本、维护 thread_id，并调用 LangGraph 工作流。",
    tags: ["python", "entrypoint", "chat", "langgraph"],
    complexity: "moderate",
    summaries: {
      Chatbot:
        "面向会话的包装器，负责准备初始状态并在多轮对话中复用 schema 元数据。",
    },
  },
  "graph.py": {
    fileSummary:
      "核心编排图，负责 SQL 与 RAG 路由、SQL 执行、失败重试以及分析输出。",
    tags: ["python", "langgraph", "workflow", "orchestration"],
    complexity: "complex",
    summaries: {
      GraphState:
        "带类型的状态容器，承载消息、schema 文本、SQL、执行结果、重试计数和分析输出。",
      build_text2sql_graph:
        "构建完整的 LangGraph 工作流，覆盖路由、SQL 生成、执行、重试与最终回答整合。",
    },
  },
  "text2sql_agent.py": {
    fileSummary:
      "ReAct 风格的 SQL 生成模块，强约束模型输出单条只读 PostgreSQL SELECT 语句。",
    tags: ["python", "llm", "sql-generation", "react"],
    complexity: "complex",
    summaries: {
      Text2sqlAgent:
        "由 LLM 驱动的代理，结合 schema、对话历史和执行错误生成或修复 SQL。",
      _extract_sql:
        "兼容解析器，用于从 markdown 代码块或纯文本中提取 SQL。",
      _extract_final_sql:
        "从模型输出的 `Final:` 段落中提取最终 SQL。",
      _resolve_sql_output:
        "优先解析 `Final:` 输出，必要时回退到旧版 SQL 提取逻辑。",
      last_user_text:
        "返回消息历史中最近一条用户输入。",
    },
  },
  "query_executor.py": {
    fileSummary:
      "只读 PostgreSQL 执行层，包含安全校验和 schema 摘要提取能力。",
    tags: ["python", "postgres", "execution", "safety"],
    complexity: "moderate",
    summaries: {
      UnsafeSQLError:
        "当静态 SQL 校验发现危险或不支持的语句时抛出。",
      QueryResult:
        "执行结果对象，包含列名、结果行和可选错误信息。",
      QueryExecutor:
        "在 PostgreSQL 上执行受保护的 SQL，并可读取 public schema 元数据供提示词使用。",
      _assert_read_only_sql:
        "在执行前拒绝空语句、非 SELECT、多语句或疑似变更型 SQL。",
    },
  },
  "analyzer_agent.py": {
    fileSummary:
      "查询后分析模块，负责写入中文 Markdown 报告，并可调用导出类技能。",
    tags: ["python", "analysis", "reporting", "skills"],
    complexity: "complex",
    summaries: {
      AnalyzerAgent:
        "由 LLM 驱动的分析器，汇总结果集并按需触发基于 skill 的导出。",
    },
  },
  "skill_runtime.py": {
    fileSummary:
      "本地 skill 加载与调用运行时，供分析器从 `skills/*/SKILL.md` 发现导出脚本。",
    tags: ["python", "skills", "runtime", "integration"],
    complexity: "moderate",
    summaries: {
      SkillToolSpec:
        "描述 skill 工具 intent、触发词和脚本路径的数据类。",
      load_skill_tools:
        "扫描 skill 目录，解析 SKILL frontmatter，并返回可执行工具定义。",
      find_skill_tool:
        "根据 intent 和用户问题选择最合适的 skill 工具。",
      invoke_skill_tool:
        "以子进程方式运行 skill 脚本，并规范化其 JSON 返回值。",
    },
  },
  "config.py": {
    fileSummary:
      "集中管理数据库、LLM、分析器和 RAG 配置的 Pydantic 设置模型。",
    tags: ["python", "config", "settings"],
    complexity: "moderate",
    summaries: {
      Settings:
        "基于环境变量的设置对象，统一承载数据库、模型、重试、分析器和 RAG 参数。",
      get_settings:
        "在整个包内复用的带缓存设置工厂。",
      resolve_llm_base_url:
        "从 provider 默认值或显式覆盖中选择 OpenAI 兼容 base URL。",
    },
  },
  "llm_client.py": {
    fileSummary:
      "共享的 LangChain OpenAI 兼容聊天模型工厂。",
    tags: ["python", "llm", "client"],
    complexity: "simple",
    summaries: {
      build_chat_model:
        "基于 provider 专属的 base URL 解析结果构建共享聊天模型客户端。",
    },
  },
  "rag_engine.py": {
    fileSummary:
      "面向知识库问答的可选检索与路由模块，用于避免某些问题直接命中实时 SQL。",
    tags: ["python", "rag", "retrieval", "routing"],
    complexity: "complex",
    summaries: {
      ApiReranker:
        "OpenAI 兼容 reranker 封装，当 API 不可用时回退到词法打分。",
      LengthSafeEmbeddings:
        "在向量化前截断长文本的 Embedding 适配器。",
      RagResult:
        "封装知识库路由结果与检索上下文的结果对象。",
      RagEngine:
        "构建并查询本地知识库，决定 SQL 或 KB 路由，并整合知识库答案。",
    },
  },
  "test_config.py": {
    fileSummary: "用于验证设置行为的小型配置测试模块。",
    tags: ["python", "test"],
    complexity: "simple",
  },
};

const NON_CODE_FILES = {
  "README.md": {
    nodeType: "document",
    summary:
      "项目总览文档，说明 Text2SQL 架构、初始化步骤和当前缺口。",
    tags: ["docs", "overview"],
    complexity: "simple",
  },
  "requirements.txt": {
    nodeType: "config",
    summary:
      "最小 Python 依赖清单，覆盖 LangGraph、LangChain、PostgreSQL 访问和可选 RAG 包。",
    tags: ["deps", "python", "config"],
    complexity: "simple",
  },
  ".env.example": {
    nodeType: "config",
    summary:
      "数据库连接、LLM 配置、分析器 skill 路径和 RAG 选项的环境变量模板。",
    tags: ["env", "config", "template"],
    complexity: "simple",
  },
};

const FLOW_NODES = [
  {
    id: "domain:text2sql-system",
    type: "domain",
    name: "Text2SQL 系统",
    summary:
      "覆盖自然语言理解、SQL 生成、执行和结果解释的业务领域。",
    tags: ["domain", "text2sql"],
    complexity: "complex",
    domainMeta: {
      entities: ["Chatbot", "LangGraph", "SQL", "RAG", "AnalyzerAgent"],
      businessRules: [
        "只允许执行只读 SQL。",
        "执行失败时会触发有上限的重试。",
        "面向未来或偏政策类的问题可以路由到知识库。",
      ],
      entryPoint: "chatbot.py",
      entryType: "cli",
    },
  },
  {
    id: "flow:text2sql-main",
    type: "flow",
    name: "主查询流程",
    summary:
      "从用户提问到 SQL 生成、执行、重试处理和分析输出的端到端流程。",
    tags: ["flow", "sql"],
    complexity: "complex",
  },
  {
    id: "flow:text2sql-rag",
    type: "flow",
    name: "知识库回退流程",
    summary:
      "面向知识型问题的替代路线，通过索引文档而不是 SQL 给出回答。",
    tags: ["flow", "rag"],
    complexity: "moderate",
  },
  {
    id: "flow:text2sql-export",
    type: "flow",
    name: "分析导出流程",
    summary:
      "分析完成后的导出流程，在用户提出需求时发现 skill 脚本并输出附加产物。",
    tags: ["flow", "skills", "reporting"],
    complexity: "moderate",
  },
  {
    id: "step:route-query",
    type: "step",
    name: "问题路由",
    summary:
      "在开始执行前，把最新问题判定为优先走 SQL 还是优先走知识库。",
    tags: ["step", "routing"],
    complexity: "moderate",
  },
  {
    id: "step:generate-sql",
    type: "step",
    name: "生成 SQL",
    summary:
      "结合 schema、对话历史和可选错误反馈，生成单条只读 PostgreSQL 查询。",
    tags: ["step", "sql-generation"],
    complexity: "complex",
  },
  {
    id: "step:execute-sql",
    type: "step",
    name: "执行 SQL",
    summary:
      "在 PostgreSQL 上执行受保护的 SQL，并收集结果行或标准化错误信息。",
    tags: ["step", "execution"],
    complexity: "moderate",
  },
  {
    id: "step:retry-or-fail",
    type: "step",
    name: "重试或失败",
    summary:
      "当执行失败时重试 SQL 生成，并在达到重试预算后停止。",
    tags: ["step", "retry"],
    complexity: "moderate",
  },
  {
    id: "step:analyze-result",
    type: "step",
    name: "分析结果",
    summary:
      "把 SQL 输出转成面向业务的中文报告，并按需导出产物。",
    tags: ["step", "analysis"],
    complexity: "moderate",
  },
  {
    id: "step:answer-with-rag",
    type: "step",
    name: "RAG 回答",
    summary:
      "当路由器决定不走 SQL 时，基于检索到的知识库片段组织回答。",
    tags: ["step", "rag"],
    complexity: "moderate",
  },
];

const MANUAL_RELATIONS = [
  ["class:chatbot.py:Chatbot", "function:graph.py:build_text2sql_graph", "depends_on", "Chatbot 在会话初始化阶段构建 LangGraph 工作流。", 0.95],
  ["class:chatbot.py:Chatbot", "class:query_executor.py:QueryExecutor", "depends_on", "Chatbot 持有 QueryExecutor，用于刷新 schema 和执行图流程。", 0.85],
  ["function:graph.py:build_text2sql_graph", "class:text2sql_agent.py:Text2sqlAgent", "depends_on", "工作流把 SQL 起草与修复交给 Text2sqlAgent。", 0.95],
  ["function:graph.py:build_text2sql_graph", "class:query_executor.py:QueryExecutor", "depends_on", "工作流通过 QueryExecutor 执行生成出的 SQL。", 0.95],
  ["function:graph.py:build_text2sql_graph", "class:analyzer_agent.py:AnalyzerAgent", "depends_on", "SQL 成功返回后由 AnalyzerAgent 负责总结。", 0.95],
  ["function:graph.py:build_text2sql_graph", "class:rag_engine.py:RagEngine", "depends_on", "路由分支会按需实例化并调用 RAG 引擎。", 0.8],
  ["function:graph.py:build_text2sql_graph", "function:text2sql_agent.py:last_user_text", "calls", "图中的路由与报告阶段都会读取最近一条用户消息。", 0.75],
  ["class:text2sql_agent.py:Text2sqlAgent", "function:llm_client.py:build_chat_model", "depends_on", "SQL 代理复用共享的 LLM 客户端工厂。", 0.9],
  ["class:analyzer_agent.py:AnalyzerAgent", "function:llm_client.py:build_chat_model", "depends_on", "AnalyzerAgent 生成报告时复用共享聊天模型工厂。", 0.9],
  ["class:analyzer_agent.py:AnalyzerAgent", "function:skill_runtime.py:load_skill_tools", "depends_on", "AnalyzerAgent 初始化时会加载项目内 skill 工具。", 0.85],
  ["class:analyzer_agent.py:AnalyzerAgent", "function:skill_runtime.py:find_skill_tool", "depends_on", "AnalyzerAgent 需要把导出 intent 映射到具体 skill 脚本。", 0.8],
  ["class:analyzer_agent.py:AnalyzerAgent", "function:skill_runtime.py:invoke_skill_tool", "depends_on", "当用户需要导出时，AnalyzerAgent 会调用 skill 子进程。", 0.8],
  ["class:query_executor.py:QueryExecutor", "function:query_executor.py:_assert_read_only_sql", "calls", "每条 SQL 在执行前都会经过静态只读校验。", 0.9],
  ["class:query_executor.py:QueryExecutor", "class:config.py:Settings", "depends_on", "数据库连接串、schema 过滤和超时参数都来自 Settings。", 0.75],
  ["function:llm_client.py:build_chat_model", "function:config.py:resolve_llm_base_url", "calls", "LLM 客户端构建时会通过配置解析 provider 对应的 base URL。", 0.85],
  ["class:rag_engine.py:RagEngine", "function:llm_client.py:build_chat_model", "depends_on", "RagEngine 复用共享聊天模型完成路由和答案整合。", 0.85],
  ["class:rag_engine.py:RagEngine", "class:config.py:Settings", "depends_on", "Embedding、rerank 和知识库路径都由 Settings 提供。", 0.8],
  ["document:README.md", "domain:text2sql-system", "documents", "README 解释了系统级别的结构和运行方式。", 0.7],
  ["config:.env.example", "class:query_executor.py:QueryExecutor", "configures", "环境变量模板中包含数据库连接配置。", 0.75],
  ["config:.env.example", "function:llm_client.py:build_chat_model", "configures", "环境变量模板中包含 provider、model 和 base URL 配置。", 0.75],
  ["config:requirements.txt", "class:rag_engine.py:RagEngine", "depends_on", "依赖清单中声明了可选的 RAG 相关包。", 0.55],
  ["domain:text2sql-system", "flow:text2sql-main", "contains_flow", "用于处理 SQL 型问题的主执行路径。", 1],
  ["domain:text2sql-system", "flow:text2sql-rag", "contains_flow", "用于处理知识库型问题的回退路径。", 1],
  ["domain:text2sql-system", "flow:text2sql-export", "contains_flow", "由本地 skill 驱动的报告导出路径。", 1],
  ["flow:text2sql-main", "step:route-query", "flow_step", "工作流从问题分类开始。", 1],
  ["flow:text2sql-main", "step:generate-sql", "flow_step", "当选择 SQL 路径后进入 SQL 生成阶段。", 1],
  ["flow:text2sql-main", "step:execute-sql", "flow_step", "生成后的 SQL 会被送往 PostgreSQL 执行。", 1],
  ["flow:text2sql-main", "step:retry-or-fail", "flow_step", "执行错误会触发重试或进入失败总结。", 1],
  ["flow:text2sql-main", "step:analyze-result", "flow_step", "成功返回的结果会被整理成面向用户的报告。", 1],
  ["flow:text2sql-rag", "step:route-query", "flow_step", "同一个路由决策也可以把流量导向知识库路径。", 1],
  ["flow:text2sql-rag", "step:answer-with-rag", "flow_step", "知识库问题通过检索文档进行回答。", 1],
  ["flow:text2sql-export", "step:analyze-result", "flow_step", "分析文本是后续导出路由的输入。", 1],
  ["flow:text2sql-export", "function:skill_runtime.py:load_skill_tools", "depends_on", "在执行导出前必须先完成 skill 发现。", 0.8],
  ["step:route-query", "file:graph.py", "related", "路由节点定义在 graph.py 的 LangGraph 编排逻辑中。", 0.8],
  ["step:generate-sql", "class:text2sql_agent.py:Text2sqlAgent", "depends_on", "Text2sqlAgent 负责 SQL 起草和修复。", 0.9],
  ["step:execute-sql", "class:query_executor.py:QueryExecutor", "depends_on", "QueryExecutor 负责真正的 PostgreSQL 调用。", 0.9],
  ["step:retry-or-fail", "file:graph.py", "related", "重试预算由 graph.py 中的条件边实现。", 0.8],
  ["step:analyze-result", "class:analyzer_agent.py:AnalyzerAgent", "depends_on", "AnalyzerAgent 把结果行转成最终报告。", 0.9],
  ["step:answer-with-rag", "class:rag_engine.py:RagEngine", "depends_on", "当跳过 SQL 时，由 RagEngine 提供知识库回答。", 0.85],
];

const LAYERS = [
  {
    id: "layer-entry",
    name: "会话入口层",
    description:
      "面向用户的会话入口以及高层工作流包装器。",
    nodeIds: [
      "class:chatbot.py:Chatbot",
      "file:chatbot.py",
      "document:README.md",
      "domain:text2sql-system",
    ],
  },
  {
    id: "layer-orchestration",
    name: "工作流编排层",
    description:
      "LangGraph 路由、SQL 生成逻辑以及知识库回退决策。",
    nodeIds: [
      "file:graph.py",
      "function:graph.py:build_text2sql_graph",
      "file:text2sql_agent.py",
      "class:text2sql_agent.py:Text2sqlAgent",
      "function:text2sql_agent.py:last_user_text",
      "file:rag_engine.py",
      "class:rag_engine.py:RagEngine",
      "flow:text2sql-main",
      "flow:text2sql-rag",
      "step:route-query",
      "step:generate-sql",
      "step:answer-with-rag",
    ],
  },
  {
    id: "layer-execution",
    name: "执行与分析层",
    description:
      "只读 SQL 执行、重试处理、报告生成以及导出钩子。",
    nodeIds: [
      "file:query_executor.py",
      "class:query_executor.py:QueryExecutor",
      "function:query_executor.py:_assert_read_only_sql",
      "file:analyzer_agent.py",
      "class:analyzer_agent.py:AnalyzerAgent",
      "file:skill_runtime.py",
      "function:skill_runtime.py:load_skill_tools",
      "function:skill_runtime.py:find_skill_tool",
      "function:skill_runtime.py:invoke_skill_tool",
      "flow:text2sql-export",
      "step:execute-sql",
      "step:retry-or-fail",
      "step:analyze-result",
    ],
  },
  {
    id: "layer-config",
    name: "配置与平台层",
    description:
      "共享设置、模型初始化、依赖清单和环境变量模板。",
    nodeIds: [
      "file:config.py",
      "class:config.py:Settings",
      "function:config.py:get_settings",
      "function:config.py:resolve_llm_base_url",
      "file:llm_client.py",
      "function:llm_client.py:build_chat_model",
      "config:requirements.txt",
      "config:.env.example",
    ],
  },
];

const TOUR = [
  {
    order: 1,
    title: "仓库整体形状",
    description:
      "这个仓库本身是名为 `text2sql` 的 Python 包，Chatbot 是会话包装器，graph.py 定义了主工作流。",
    nodeIds: ["document:README.md", "class:chatbot.py:Chatbot", "file:graph.py"],
    languageLesson:
      "先从 Chatbot 进入，再沿着 `build_text2sql_graph` 理解问题如何变成 SQL 或知识库回答。",
  },
  {
    order: 2,
    title: "先决定走哪条路",
    description:
      "工作流会先判断这次请求更适合走 SQL 还是知识库路径。",
    nodeIds: ["step:route-query", "class:rag_engine.py:RagEngine", "function:text2sql_agent.py:last_user_text"],
    languageLesson:
      "面向未来或偏政策的问题会被有意从实时 SQL 路径切到 RAG 路径。",
  },
  {
    order: 3,
    title: "生成安全 SQL",
    description:
      "Text2sqlAgent 使用 ReAct 风格提示词和执行反馈修复 SQL，同时始终保持只读约束。",
    nodeIds: ["class:text2sql_agent.py:Text2sqlAgent", "step:generate-sql", "class:query_executor.py:QueryExecutor"],
    languageLesson:
      "重点看 text2sql_agent.py 里的 `Final:` 输出契约，以及 query_executor.py 里的静态安全校验。",
  },
  {
    order: 4,
    title: "重试并生成分析",
    description:
      "执行失败会在配置上限内重试，成功结果会被整理成中文分析报告。",
    nodeIds: ["step:execute-sql", "step:retry-or-fail", "class:analyzer_agent.py:AnalyzerAgent"],
    languageLesson:
      "重试预算在 Settings 中配置，而最终 Markdown 报告和落盘产物由 AnalyzerAgent 负责。",
  },
  {
    order: 5,
    title: "通过 Skills 扩展",
    description:
      "AnalyzerAgent 可以发现项目内 skill，并在报告生成后运行导出脚本。",
    nodeIds: ["flow:text2sql-export", "file:skill_runtime.py", "function:skill_runtime.py:invoke_skill_tool"],
    languageLesson:
      "skill_runtime.py 是分析文本与 SKILL.md 所定义外部脚本之间的桥梁。",
  },
];

function posixPath(value) {
  return value.split(path.sep).join("/");
}

function readText(projectRoot, relativePath) {
  return fs.readFileSync(path.join(projectRoot, relativePath), "utf-8");
}

function exists(projectRoot, relativePath) {
  return fs.existsSync(path.join(projectRoot, relativePath));
}

function resolveGitHash(projectRoot) {
  try {
    return execSync("git rev-parse HEAD", {
      cwd: projectRoot,
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString("utf-8")
      .trim();
  } catch {
    return "";
  }
}

function collectFiles(projectRoot, ignoreFilter) {
  const results = [];
  const walk = (relativeDir) => {
    const absoluteDir = path.join(projectRoot, relativeDir);
    for (const entry of fs.readdirSync(absoluteDir, { withFileTypes: true })) {
      const relativePath = relativeDir
        ? posixPath(path.join(relativeDir, entry.name))
        : entry.name;
      if (entry.isDirectory()) {
        if (ignoreFilter.isIgnored(`${relativePath}/`)) continue;
        walk(relativePath);
        continue;
      }
      if (ignoreFilter.isIgnored(relativePath)) continue;
      results.push(relativePath);
    }
  };
  walk("");
  return results.sort((a, b) => a.localeCompare(b));
}

function deriveFileMeta(relativePath, analysis) {
  const override = CODE_OVERRIDES[relativePath] ?? {};
  const summaries = { ...(override.summaries ?? {}) };

  for (const fn of analysis.functions ?? []) {
    if (!summaries[fn.name]) {
      summaries[fn.name] = `位于 ${relativePath} 的顶层辅助函数。`;
    }
  }
  for (const cls of analysis.classes ?? []) {
    if (!summaries[cls.name]) {
      summaries[cls.name] = `定义在 ${relativePath} 中的主要类。`;
    }
  }

  const classCount = analysis.classes?.length ?? 0;
  const functionCount = analysis.functions?.length ?? 0;
  return {
    fileSummary:
      override.fileSummary ??
      `Python 模块 ${relativePath}，包含 ${classCount} 个类和 ${functionCount} 个顶层函数。`,
    tags: override.tags ?? ["python"],
    complexity: override.complexity ?? "moderate",
    summaries,
  };
}

function resolveInternalImports(relativePath, content, knownFiles) {
  const targets = new Set();
  const regexes = [
    /^\s*from\s+text2sql\.([A-Za-z0-9_]+)\s+import\b/gm,
    /^\s*import\s+text2sql\.([A-Za-z0-9_]+)/gm,
    /^\s*from\s+\.([A-Za-z0-9_]+)\s+import\b/gm,
  ];

  for (const regex of regexes) {
    for (const match of content.matchAll(regex)) {
      const candidate = `${match[1]}.py`;
      if (knownFiles.has(candidate)) {
        targets.add(candidate);
      }
    }
  }

  targets.delete(relativePath);
  return [...targets].sort((a, b) => a.localeCompare(b));
}

function edgeKey(edge) {
  return `${edge.type}|${edge.source}|${edge.target}`;
}

function addManualNode(graph, node) {
  if (graph.nodes.some((item) => item.id === node.id)) return;
  graph.nodes.push(node);
}

function addEdge(graph, nodeIds, seen, source, target, type, description, weight) {
  if (!nodeIds.has(source) || !nodeIds.has(target)) return;
  const edge = {
    source,
    target,
    type,
    direction: "forward",
    description,
    weight,
  };
  const key = edgeKey(edge);
  if (seen.has(key)) return;
  seen.add(key);
  graph.edges.push(edge);
}

async function loadCore() {
  const home = process.env.HOME ?? "";
  const candidates = [
    path.join(home, ".understand-anything-plugin", "packages", "core", "dist", "index.js"),
    path.join(home, ".understand-anything", "repo", "understand-anything-plugin", "packages", "core", "dist", "index.js"),
    path.join(PROJECT_ROOT, ".understand-anything", "understand-anything-plugin", "packages", "core", "dist", "index.js"),
  ];

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return import(pathToFileURL(candidate).href);
    }
  }
  throw new Error("Could not locate @understand-anything/core dist build.");
}

async function buildGraph() {
  const core = await loadCore();
  const {
    TreeSitterPlugin,
    PluginRegistry,
    builtinLanguageConfigs,
    registerAllParsers,
    GraphBuilder,
    saveGraph,
    saveMeta,
    saveConfig,
    loadGraph,
    createIgnoreFilter,
    validateGraph,
  } = core;

  const tsPlugin = new TreeSitterPlugin(
    builtinLanguageConfigs.filter((config) => config.treeSitter),
  );
  await tsPlugin.init();

  const registry = new PluginRegistry();
  registry.register(tsPlugin);
  registerAllParsers(registry);

  const ignoreFilter = createIgnoreFilter(PROJECT_ROOT);
  const scannedFiles = collectFiles(PROJECT_ROOT, ignoreFilter);
  const pythonFiles = scannedFiles.filter(
    (file) => file.endsWith(".py") && !path.basename(file).startsWith("test_"),
  );
  const nonCodeFiles = Object.keys(NON_CODE_FILES).filter((file) =>
    exists(PROJECT_ROOT, file),
  );
  const knownFiles = new Set(scannedFiles);
  const gitHash = resolveGitHash(PROJECT_ROOT);
  const builder = new GraphBuilder(path.basename(PROJECT_ROOT), gitHash);

  for (const relativePath of pythonFiles) {
    const content = readText(PROJECT_ROOT, relativePath);
    const analysis =
      registry.analyzeFile(relativePath, content) ?? {
        functions: [],
        classes: [],
        imports: [],
        exports: [],
      };
    builder.addFileWithAnalysis(
      relativePath,
      analysis,
      deriveFileMeta(relativePath, analysis),
    );
  }

  for (const relativePath of nonCodeFiles) {
    builder.addNonCodeFile(relativePath, NON_CODE_FILES[relativePath]);
  }

  for (const relativePath of pythonFiles) {
    const content = readText(PROJECT_ROOT, relativePath);
    for (const imported of resolveInternalImports(relativePath, content, knownFiles)) {
      builder.addImportEdge(relativePath, imported);
    }
  }

  const graph = builder.build();
  graph.project.frameworks = ["LangGraph", "LangChain", "PostgreSQL"];
  graph.project.description =
    "一个 text2sql 原型包，支持把问题路由到 SQL 或 RAG，执行受保护的 PostgreSQL 查询，并生成中文分析报告。";

  for (const node of FLOW_NODES) {
    addManualNode(graph, node);
  }

  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  const seenEdges = new Set(graph.edges.map(edgeKey));
  for (const relation of MANUAL_RELATIONS) {
    addEdge(
      graph,
      nodeIds,
      seenEdges,
      relation[0],
      relation[1],
      relation[2],
      relation[3],
      relation[4],
    );
  }

  graph.layers = LAYERS.map((layer) => ({
    ...layer,
    nodeIds: layer.nodeIds.filter((id) => nodeIds.has(id)),
  }));
  graph.tour = TOUR.map((step) => ({
    ...step,
    nodeIds: step.nodeIds.filter((id) => nodeIds.has(id)),
  }));

  const validation = validateGraph(graph);
  if (!validation.success || !validation.data) {
    throw new Error(validation.fatal ?? "Graph validation failed");
  }

  const analyzedFileCount = pythonFiles.length + nonCodeFiles.length;

  saveGraph(PROJECT_ROOT, graph);
  saveMeta(PROJECT_ROOT, {
    lastAnalyzedAt: graph.project.analyzedAt,
    gitCommitHash: gitHash,
    version: graph.version,
    analyzedFiles: analyzedFileCount,
  });
  saveConfig(PROJECT_ROOT, {
    autoUpdate: false,
    outputLanguage: "zh",
  });

  const loaded = loadGraph(PROJECT_ROOT);
  if (!loaded) {
    throw new Error("Graph load returned null after save.");
  }

  const result = {
    projectRoot: PROJECT_ROOT,
    graphPath: path.join(PROJECT_ROOT, ".understand-anything", "knowledge-graph.json"),
    metaPath: path.join(PROJECT_ROOT, ".understand-anything", "meta.json"),
    configPath: path.join(PROJECT_ROOT, ".understand-anything", "config.json"),
    nodeCount: loaded.nodes.length,
    edgeCount: loaded.edges.length,
    layerCount: loaded.layers.length,
    tourStepCount: loaded.tour.length,
    analyzedFiles: analyzedFileCount,
    validationIssues: validation.issues.length,
  };

  console.log(JSON.stringify(result, null, 2));
}

buildGraph().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exit(1);
});
