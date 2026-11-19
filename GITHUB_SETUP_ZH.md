# GitHub创建步骤

本地仓库名：`tsad-calibration-study`，目标可见性：Public。

不要把GitHub personal access token粘贴到聊天、README、脚本或Git历史中。

## 先核对数据权利

阅读并完成：

```text
DATA_RIGHTS_REVIEW.csv
THIRD_PARTY_NOTICES.md
RELEASE_MANIFEST.json
```

MIT仅覆盖项目自写代码。对于派生score、labels、HAI覆盖修正和第三方组件，作者必须确认公开再分发的权利。

## 使用交互式GitHub登录

在自己的终端运行：

```sh
brew install gh                 # 若尚未安装GitHub CLI
gh auth login                   # 浏览器或device code交互式授权；不要在聊天中分享token
cd /path/to/tsad-calibration-study
CONFIRM_DERIVED_DATA_RIGHTS=YES ./PUBLISH_TO_GITHUB.sh
```

脚本会创建Public仓库、首次push并推送`v0.1.0-paa-candidate`标签。它在未设置确认变量时会失败，不会创建远程仓库。

## 创建后

1. 打开GitHub网页，查看文件、LICENSE、README和tag。
2. 在未登录窗口下载或clone，运行README的复现命令。
3. 创建Zenodo新版本draft；上传固定版本归档或将GitHub release关联到Zenodo。
4. 只有在Zenodo版本已实际发布、下载可用、DOI/commit/tag已核对后，才在PAA稿件的数据代码声明和投稿信中写入URL及DOI。
