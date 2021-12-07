# Food_cls Baseline

A simple baseline for SIGS_Big_Data_ML_Exam_2021.

https://www.kaggle.com/t/b7ed697207f0401b94a1f5c49c559d68

# Environment
- python 3.6
- torch 1.5.1
- torchvision 0.6.0
- tqdm

必须用 `GPU` 跑 Q.Q (3 min per epoch on Tesla T4 8GB Memory)

# Download

下载数据到指定路径 `./data/food/`,将三个文件夹分别移动到:

- `./data/food/train`
- `./data/food/val`
- `./data/food/test`

# Prepare

生成索引文件，创建数据集：

`python prepare.py --src ./data/food/train --out ./data/food/train.txt`

`python prepare.py --src ./data/food/val  --out ./data/food/val.txt`

修改 `dataset.py` 的 `107-108` 行为你的指定路径

# Hyper-parameter

修改 `config.py` 的超参数为你需要的值

`root`修改为你的项目本地路径

# Train

`CUDA_VISIBLE_DEVICES=0 python train.py`

# Inferance

这部分代码请同学们自己实现

功能为用训练好的模型测试 `./data/food/test` 路径下的所有图片，并生成 `submission.txt` 文件

请注意提交格式

# Tips

1. 这只是个baseline，不要求一定使用这个代码
2. 遇到问题及时与助教沟通，或者提 `issues`
3. 请维护好自己的 `git commit` 记录，尽量在每次 `commit` 时都写明自己的具体工作
