# 在实际训练设备上导出证据（不跑模型）

用户已明确：模型在另一台设备训练。本机只做分数复算、代码静态审查与论文构建。本地发现的数据副本不作为训练设备当时输入的证明。

**更新**：作者已确认训练数据全部来自核查过的官方公开源，来源问题已关闭。这个导出工具现在是可选的复现记录工具，不是继续修稿的前置条件，也不要求重新证明下载来源。

将本目录复制到**当时跑实验的设备**，在其完整项目根目录执行：

```powershell
python export_training_evidence.py --project-root "C:\实际路径\tsad-paper" --output "D:\exports\MTSAD_training_evidence" --dataset SKAB --dataset HAI --dataset SMD
```

路径请改为实际值。输出目录必须不存在，脚本拒绝覆盖。只需要Python标准库，**不需要导入torch、不训练、不推理、不联网、不安装依赖**。

## 会导出

- 指定9个训练/数据/评分脚本的当前副本与hash；
- TimesNet/MEMTO/CATCH相关run的白名单配置、指标、每阶段epoch记录和数值训练日志；
- 指定dataset目录中CSV/NPY等文件的文件名、字节数、SHA256；
- 已安装包的版本元数据（不会导入或执行这些包）。

## 不会导出

原始数据内容、模型权重、环境变量、API密钥、`.env`、用户主目录、完整环境清单或任意系统日志。源代码本身可能含专有信息，分享前请人工检查。脚本不自动上传。

## 最重要的限制

**现在存在的文件不等于过去确实使用过的文件。**必须结合当时日志、run fingerprint、版本记录或历史数据manifest，才能把当前文件绑定到2026-11-06那30个run。脚本默认将`historical_usage_verified`设为false，不能靠人工改成true来补证据。

请另附一份简短人工说明：

1. SKAB运行时是否有`data/SKAB/anomaly-free/`，是否改变过数据路径或loader？
2. `models_recent.py`与`run_recent.py`在实验后是否修改过？能否恢复当时版本？
3. HAI原始feature和label文件是否有时间戳？如何对齐？
4. CATCH/MEMTO是否有保存权重和官方仓库commit？这里只询问是否有，不要求上传权重。

导出包回来后可以进一步检查历史fingerprint、数据文件身份与加载分支。若有证据缺口，将如实保留；不会为了投稿把推测写成事实。
