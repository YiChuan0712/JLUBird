from pydub import AudioSegment
import wave

# 将时间转换为毫秒
def time_to_ms(hour=0, minute=0, second=0, millisecond=0):

    if isinstance(hour, int) is False:
        return "'hour' should be of type 'int'"
    if hour < 0:
        return "'hour' should be in the range of [0, +∞)"

    if isinstance(minute, int) is False:
        return "'minute' should be of type 'int'"
    if minute < 0 or minute >= 60:
        return "'minute' should be in the range of [0, 59]"

    if isinstance(second, int) is False:
        return "'second' should be of type 'int'"
    if second < 0 or second >= 60:
        return "'second' should be in the range of [0, 59]"

    if isinstance(millisecond, int) is False:
        return "'millisecond' should be of type 'int'"
    if millisecond < 0 or millisecond > 999:
        return "'millisecond' should be in the range of [0, 59]"

    hour = hour * 60 * 60
    minute = minute * 60
    total_sec = hour + minute + second
    total_ms = total_sec * 1000
    total_ms += millisecond

    return total_ms

# 将毫秒转换为时间
def ms_to_time(total_ms=0):

    if isinstance(total_ms, int) is False:
        return "'total_ms' should be of type 'int'"
    if total_ms < 0:
        return "'total_ms' should be in the range of [0, +∞)"

    total_sec = int(total_ms / 1000)
    millisecond = int(total_ms - total_sec * 1000)
    hour = int(total_sec / 3600)
    total_sec = total_sec - hour * 3600
    minute = int(total_sec / 60)
    second = int(total_sec - minute * 60)

    return hour, minute, second, millisecond

# 切割并保存.wav
def split_and_save(input_file, output_file, begin=0, end=0):

    sound = AudioSegment.from_wav(input_file)
    duration = sound.duration_seconds * 1000  # 音频时长 ms

    if 0 <= begin < end <= duration is False:
        return "make sure 0 <= begin < end <= duration"

    begin_tuple = ms_to_time(begin)
    end_tuple = ms_to_time(end)

    section1 = "(" + str(begin_tuple[0]).zfill(2) + "-"\
                   + str(begin_tuple[1]).zfill(2) + "-"\
                   + str(begin_tuple[2]).zfill(2) + "-"\
                   + str(begin_tuple[3]).zfill(3)\
                   + ")"\
                   + "(" + str(end_tuple[0]).zfill(2) + "-" \
                   + str(end_tuple[1]).zfill(2) + "-" \
                   + str(end_tuple[2]).zfill(2) + "-" \
                   + str(end_tuple[3]).zfill(3) \
                   + ")"
    output_file = output_file.rsplit('.',1)[0] + section1 + '.wav'

    cut_wav = sound[begin:end]  # 以毫秒为单位截取[begin, end]区间的音频
    cut_wav.export(output_file, format='wav')  # 存储新的wav文件

    print(output_file)

# 合并并保存.wav
def merge_and_save(input_files, output_file):
    data= []
    for infile in input_files:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()

    output = wave.open(output_file, 'wb')
    output.setparams(data[0][0])
    output.writeframes(data[0][1])
    output.writeframes(data[1][1])
    output.close()

#print(time_to_ms(0,20,0,0))

# split_and_save(r"D:\birdsound\test211112.wav", r"D:\birdsound\12324.wav", time_to_ms(0,10,0,0), time_to_ms(0,30,0,0))
# r"D:\birdsound\short(00-20-00-000)(00-30-00-000).wav"