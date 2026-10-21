
def read_file(path:str):
    """
    读取一个文本文件的内容
    :param path: 要读取的文件路径
    :return: 返回读取的内容
    """
    with open(path,"r",encoding = "utf-8") as f:
        return f.read()

def write_file(path:str,content:str):
    """
    写入一些内容到某个文本文件中
    :param path: 要写入的文件路径
    :param content: 要写入的内容
    """
    if isinstance(content,str):
        with open(path,"a",encoding = "utf-8") as f:
            f.write(content)
        return f"已经写入文件:{path}"
    else :
        return f"写入的内容不是字符串，不合法"