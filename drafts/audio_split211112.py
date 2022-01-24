"""
audio_split211112.py by Yichuan
此文件用于分割并保存.wav格式的音频

time_to_ms(hour, minute, sec) 将时分秒 转化为毫秒
    hour minute sec 请填入int
    特别注意!!! hour请填入 小于100 的int!!!

ms_to_time(total_ms) 将毫秒 转化为时分秒
    返回值为int元组
    (hour, minute, sec)

split_save_wav(filename, begin, end) 将音频切分并保存
    注意 在使用此函数之前，请先按照格式配置PATH
    filename格式见 测试程序段A 不添加.wav后缀
    begin 和 end 的单位为毫秒 请用time_to_ms 进行转换
    生成文件名 格式为 源文件名-起始时间-中止时间
    例如我要截取 test.wav 的1:1:1 到2:2:2
    那么生成的文件名字会是 test-010101-020202.wav

"""

from pydub import AudioSegment

# 切割后文件的保存路径 文件夹即可 例r"D:\birdsound"
PATH = r'D:\birdsound'


# 将时间转换为毫秒
def time_to_ms(hour, minute, sec):
    hour = hour * 60 * 60
    minute = minute * 60
    # sec = sec
    total_sec = hour + minute + sec
    total_ms = total_sec * 1000
    return total_ms


# 将毫秒转换为时间
def ms_to_time(total_ms):
    total_sec = total_ms / 1000
    hour = int(total_sec / 3600)
    total_sec = total_sec - hour * 3600
    minute = int(total_sec / 60)
    sec = int(total_sec - minute * 60)
    return hour, minute, sec

# 测试代码
# begin = time_to_ms(0, 25, 0)
# print(begin)
# print(ms_to_time(begin))


# wav分割函数
def split_save_wav(filename, begin, end):
    sound = AudioSegment.from_wav(PATH+ '\\' +filename+".wav")
    duration = sound.duration_seconds * 1000  # 音频时长（ms）
    if 0 <= begin < end <= duration is False:
        print("split_save_wav - errors in begin and end")
        return
    cut_wav = sound[begin:end]  # 以毫秒为单位截取[begin, end]区间的音频
    savefilename = filename \
                   +"-"+str(ms_to_time(begin)[0]).zfill(2)\
                   +str(ms_to_time(begin)[1]).zfill(2)\
                   +str(ms_to_time(begin)[2]).zfill(2)\
                   +"-"+str(ms_to_time(end)[0]).zfill(2)\
                   +str(ms_to_time(end)[1]).zfill(2)\
                   +str(ms_to_time(end)[2]).zfill(2)
    cut_wav.export(PATH + '\\' + savefilename + '.wav', format='wav')  # 存储新的wav文件
    print("已导出到"+PATH + '\\' + savefilename + '.wav')


"""
下面是测试程序段A
从时长为一小时的音频中截取一分钟
我选取了一段有鸟叫的00/23/00 - 00/24/00
"""

# begin = time_to_ms(0, 23, 0)
# end = time_to_ms(0, 24, 0)
# split_save_wav("test211112", begin, end)

"""
下面是测试程序段B
从时长为一小时的音频中截取一分钟
我选取了一段没有鸟叫的00/00/00 - 00/01/00
"""

# begin = time_to_ms(0, 0, 0)
# end = time_to_ms(0, 1, 0)
# split_save_wav("test211112", begin, end)


"""
下面是测试程序段C
从时长为一小时的音频中截取十秒
我选取了一段没有鸟叫的00/01/00 - 00/01/10
"""

# begin = time_to_ms(0, 1, 0)
# end = time_to_ms(0, 1, 10)
# split_save_wav("test211112", begin, end)

