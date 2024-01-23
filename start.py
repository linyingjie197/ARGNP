import os  #提供与操作系统交互的功能，例如文件和目录操作。
import sys  #提供与Python解释器交互的功能，例如获取命令行参数。
import hydra  #用于简化配置和应用程序启动的库。
from hydra.utils import get_original_cwd, to_absolute_path
#前者用于获取原始工作目录的路径。后者用于将相对路径转换为绝对路径。
from pathlib import Path  #用于处理文件系统路径。

#用于标识一个函数作为Hydra应用程序的入口点。它允许使用Hydra进行配置管理，并提供了一些额外的功能和工具。
#传递了两个参数：`config_path`和`config_name`。`config_path`参数指定了配置文件的路径。`config_name`参数指定了配置文件的名称。
@hydra.main(config_path='conf', config_name = 'start')
def start(config):

    print(config)
    
    #执行一个Shell命令，将字符串输出到控制台。这个命令是通过调用操作系统的命令行来实现的。
    os.system("echo 'start searching architectures using expand strategy!!!'")
    nb_nodes = 2
    nb_fixed_nodes = 0
    expand_from = ""  
    for i in range(1, config.repeats + 1):
        os.system(f"echo '** repeat {i}'")
        epochs = config.warmup_dec_epoch+((nb_nodes-nb_fixed_nodes)*2-1)*config.decision_freq+1
        arch_save = Path(config.save).joinpath(f"repeat{i}")  #指定了保存结果的路径
        cmd = f"""cd {get_original_cwd()}&&CUDA_VISIBLE_DEVICES={config.gpu} python search.py \
            basic.nb_nodes={nb_nodes - nb_fixed_nodes} \
            ds={config.data} \
            optimizer=search_optimizer \
            basic.search_epochs={epochs} \
            ds.expand_from={expand_from} \
            ds.arch_save={arch_save}
        """  #调用`search.py`脚本并传递一些参数。
        print(cmd)
        os.system(cmd)
        nb_fixed_nodes = nb_nodes
        nb_nodes *= 2
        #构建一个文件路径，并将其赋值给`expand_from`变量。使用`arch_save.joinpath()`方法将`arch_save`路径和后续的路径片段连接起来。
        expand_from = arch_save.joinpath(config.data, str(epochs-1), "cell_geno.txt")

#当一个Python脚本文件被直接运行时，`__name__`的值会被设置为`'__main__'`，则执行`start()`函数。而当一个Python模块被作为其他脚本文件的模块导入时，`__name__`的值会是模块的名称。
#可以让这个脚本文件既可以作为一个独立的可执行程序运行，又可以作为一个模块被其他脚本文件导入和调用。
if __name__ == '__main__':
    start()