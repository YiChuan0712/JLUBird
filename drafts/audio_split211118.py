"""
audio_split211118.py by Yichuan
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
import wave

WHITE_BEGIN = "\033[1;30m"
WHITE_END = "\033[0m"
YELLOW_BEGIN = "\033[1;33m"
YELLOW_END = "\033[0m"

# 将时间转换为毫秒
def time_to_ms(hour=0, minute=0, second=0, millisecond=0, track_info=True):

    if track_info:
        print(WHITE_BEGIN +
              "time_to_ms("+str(hour)+", "+str(minute)+", "+str(second)+", "+str(millisecond)+") -> "
              + WHITE_END
          , end="")

    if isinstance(hour, int) is False:
        if track_info: print(YELLOW_BEGIN+"'hour' should be of type 'int'"+YELLOW_END)
        return "'hour' should be of type 'int'"
    if hour < 0:
        if track_info: print(YELLOW_BEGIN+"'hour' should be in the range of [0, +∞)"+YELLOW_END)
        return "'hour' should be in the range of [0, +∞)"

    if isinstance(minute, int) is False:
        if track_info: print(YELLOW_BEGIN+"'minute' should be of type 'int'"+YELLOW_END)
        return "'minute' should be of type 'int'"
    if minute < 0 or minute >= 60:
        if track_info: print(YELLOW_BEGIN+"'minute' should be in the range of [0, 59]"+YELLOW_END)
        return "'minute' should be in the range of [0, 59]"

    if isinstance(second, int) is False:
        if track_info: print(YELLOW_BEGIN+"'second' should be of type 'int'"+YELLOW_END)
        return "'second' should be of type 'int'"
    if second < 0 or second >= 60:
        if track_info: print(YELLOW_BEGIN+"'second' should be in the range of [0, 59]"+YELLOW_END)
        return "'second' should be in the range of [0, 59]"

    if isinstance(millisecond, int) is False:
        if track_info: print(YELLOW_BEGIN+"'millisecond' should be of type 'int'"+YELLOW_END)
        return "'millisecond' should be of type 'int'"
    if millisecond < 0 or millisecond > 999:
        if track_info: print(YELLOW_BEGIN+"'millisecond' should be in the range of [0, 999]"+YELLOW_END)
        return "'millisecond' should be in the range of [0, 59]"

    hour = hour * 60 * 60
    minute = minute * 60
    total_sec = hour + minute + second
    total_ms = total_sec * 1000
    total_ms += millisecond

    if track_info:
        print(WHITE_BEGIN + "return "+str(int(total_ms)) + WHITE_END)

    return total_ms

# 将毫秒转换为时间
def ms_to_time(total_ms=0, track_info=True):

    if track_info:
        print(WHITE_BEGIN + "ms_to_time("+str(total_ms)+") -> " + WHITE_END
          , end="")

    if isinstance(total_ms, int) is False:
        if track_info: print(YELLOW_BEGIN+"'total_ms' should be of type 'int'"+YELLOW_END)
        return "'total_ms' should be of type 'int'"
    if total_ms < 0:
        if track_info: print(YELLOW_BEGIN+"'total_ms' should be in the range of [0, +∞)"+YELLOW_END)
        return "'total_ms' should be in the range of [0, +∞)"

    total_sec = int(total_ms / 1000)
    millisecond = int(total_ms - total_sec * 1000)
    hour = int(total_sec / 3600)
    total_sec = total_sec - hour * 3600
    minute = int(total_sec / 60)
    second = int(total_sec - minute * 60)

    if track_info:
        print(WHITE_BEGIN+
              "return ("+str(int(hour))+", "+str(int(minute))+", "+str(int(second))+", "+str(int(millisecond))+")"
              +WHITE_END
          , end="")

    return hour, minute, second, millisecond

# 切割并保存.wav
def split_and_save(input_file, begin, end, output_file, track_info=True):

    if track_info:
        print(WHITE_BEGIN + "split_and_save(...) -> " + WHITE_END
          , end="")

    sound = AudioSegment.from_wav(input_file)
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
    cut_wav.export(output_file)  # 存储新的wav文件
    print("已导出到"+PATH + '\\' + savefilename + '.wav')

print(str(3).zfill(2))

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

