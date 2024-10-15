# -*- coding: utf-8 -*-
# @Time : 2022/7/6 15:36 
# @Author : XXXXXX
# @File : Crawl_picture
# @Project: Crawl_picture
# @Software: PyCharm
# 源码是在CSDN上找的一个大佬的，时间很长了
# 注意：里面的一些信息请自行修改！

import re  # 导入正则表达式模块，用于处理字符串
import requests  # 导入requests模块，用于发送HTTP请求
from urllib import error  # 从urllib中导入错误处理模块
from bs4 import BeautifulSoup  # 从bs4导入BeautifulSoup，用于解析HTML
import os  # 导入os模块，用于处理文件和目录

# 定义全局变量
num = 0  # 已下载的图片数量
numPicture = 0  # 用户想要下载的图片总数量
file = ''  # 存储图片的文件夹路径
List = []  # 存储找到的图片URL列表

# 查找图片URL的函数
def Find(url, A):
    global List  # 使用全局变量List
    print('正在检测图片总数，请稍等.....')  # 提示用户正在检测图片数量
    t = 0  # 页码计数器
    s = 0  # 图片数量计数器

    while t < 1000:  # 限制最多查找1000页
        Url = url + str(t)  # 生成当前页的URL
        try:
            # 尝试发送请求获取当前页内容
            Result = A.get(Url, timeout=7, allow_redirects=False)
        except BaseException:
            t = t + 60  # 如果请求失败，增加页码并继续
            continue
        else:
            result = Result.text  # 获取网页内容
            # 使用正则表达式找到图片的URL
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)
            s += len(pic_url)  # 统计找到的图片数量
            if len(pic_url) == 0:  # 如果没有找到图片
                break  # 结束循环
            else:
                List.append(pic_url)  # 将找到的图片URL添加到List中
                t = t + 60  # 增加页码
    return s  # 返回找到的图片数量

# 获取相关推荐的函数
def recommend(url):
    Re = []  # 存储推荐的关键词
    try:
        # 尝试发送请求获取相关推荐
        html = requests.get(url, allow_redirects=False)
    except error.HTTPError as e:
        return  # 如果请求失败，返回空
    else:
        html.encoding = 'utf-8'  # 设置编码
        bsObj = BeautifulSoup(html.text, 'html.parser')  # 解析HTML
        div = bsObj.find('div', id='topRS')  # 找到推荐关键词的div
        if div is not None:
            listA = div.findAll('a')  # 找到所有的链接
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())  # 添加推荐关键词到列表
        return Re  # 返回推荐关键词列表

# 下载图片的函数
def dowmloadPicture(html, keyword):
    global num  # 使用全局变量num
    # 使用正则表达式找到图片的URL
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')  # 提示用户开始下载
    for each in pic_url:  # 遍历找到的每个图片URL
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))  # 提示正在下载的图片信息
        try:
            if each is not None:
                # 尝试下载图片
                pic = requests.get(each, timeout=7)
            else:
                continue  # 如果URL为空，继续下一个
        except BaseException:
            print('错误，当前图片无法下载')  # 下载失败的提示
            continue
        else:
            # 构建图片保存路径
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')  # 以二进制写入模式打开文件
            fp.write(pic.content)  # 写入图片内容
            fp.close()  # 关闭文件
            num += 1  # 增加已下载图片计数
        if num >= numPicture:  # 如果达到用户要求的下载数量
            return  # 结束下载

# 主函数入口
if __name__ == '__main__':
    ##############################
    # 设置请求头，模拟浏览器访问
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }

    A = requests.Session()  # 创建一个会话对象
    A.headers = headers  # 设置请求头
    ###############################
    
    # 提示用户输入搜索关键词
    word = input("请输入搜索关键词(可以是人名，地名等): ")
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='

    # 查找图片总数
    tot = Find(url, A)
    Recommend = recommend(url)  # 获取相关推荐
    print('经过检测%s类图片共有%d张' % (word, tot))  # 输出找到的图片数量
    numPicture = int(input('请输入想要下载的图片数量 '))  # 提示用户输入下载数量
    file = input('请建立一个存储图片的文件夹，输入文件夹名称即可')  # 提示用户输入存储文件夹名称
    y = os.path.exists(file)  # 检查文件夹是否存在
    if y == 1:
        print('该文件已存在，请重新输入')  # 如果文件夹已存在，提示用户重新输入
        file = input('请建立一个存储图片的文件夹，)输入文件夹名称即可')
        os.mkdir(file)  # 创建新文件夹
    else:
        os.mkdir(file)  # 创建文件夹
    t = 0  # 图片计数器
    tmp = url  # 保存原始URL以便后续使用
    while t < numPicture:  # 循环直到下载所需数量
        try:
            url = tmp + str(t)  # 生成当前页的URL
            result = A.get(url, timeout=10, allow_redirects=False)  # 获取当前页内容

        except error.HTTPError as e:
            print('网络错误，请调整网络后重试')  # 网络错误提示
            t = t + 60  # 页码增加
        else:
            dowmloadPicture(result.text, word)  # 下载图片
            t = t + 60  # 页码增加
    print('当前搜索结束，感谢使用')  # 提示用户搜索结束
    print('猜你喜欢')  # 提示用户相关推荐

    for re in Recommend:  # 输出推荐关键词
        print(re, end='  ')  # 每个推荐词之间用空格分隔
