# 作者发布与 GitHub 上传指南

## 已完成的模型配置

本包包含真实的结构、词表、特殊 token 和推理参数。权重来源为 Hugging Face loveCloud/AutoTCR 的 AutoTCR.pth。原服务器 pretrained_path 已从公开推理配置中移除。用户不需要修改配置即可下载并运行。

## 上传代码

在 GitHub 创建空仓库 AutoTCR。在解压后包含 pyproject.toml 的目录运行：

```bash
git init -b main
git add .
git status
git commit -m "Release AutoTCR inference software"
git remote add origin https://github.com/YOUR_USERNAME/AutoTCR.git
git push -u origin main
```

替换 YOUR_USERNAME。不要另行自动创建 README，以免两个初始提交冲突。此仓库的大权重由 .gitignore 排除，已改为 Hugging Face 下载流程，Git LFS 不再需要。

上传完成后可创建 v1.1.0 Release，为论文记录软件版本或 commit。本文档中的命令供作者自行执行；本次交付没有向远端仓库推送。

## 从干净环境验证

```bash
git clone https://github.com/YOUR_USERNAME/AutoTCR.git
cd AutoTCR
python -m pip install -e ".[runtime]"
autotcr download-model --output-dir models/autotcr
autotcr validate-config --model-dir models/autotcr --device cpu
autotcr repertoire --model-dir models/autotcr \
  --input examples/repertoire.csv --summary-output outputs/summary.csv --device cpu
```

## 模型版本与可复现性

代码固定权重的 Hugging Face commit，并验证 SHA-256。后续如更新模型，需要同时更新对应资产、hub.py 发布常量、测试和版本号，避免 main 分支变化影响论文复现。

Hugging Face 当前公开文件只有权重，匹配的小型配置已经随代码打包。你也可以将 config.json、vocab.txt、tokenizer_config.json、inference_config.yaml 与模型说明另外上传到 Hugging Face，方便使用者查看。它们位于 src/autotcr/assets/，无需改变词表顺序。

## 发表前补齐信息

核对 CITATION.cff 作者列表与论文 DOI；作者应明确原创代码和模型权重许可证。分词器的 Apache-2.0 版权说明和许可证已经保留。发布版本不应混入真实患者文件、训练全集或本地输出。
