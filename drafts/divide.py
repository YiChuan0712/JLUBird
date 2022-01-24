from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import eyed3
import hmmlearn
import plotly
import tqdm

# [Fs, x] = aIO.read_audio_file(r"D:\birdsound\merge2.wav")
#
# segments = aS.silence_removal(x, Fs, 5, 0.5, 0.9, 0.2, plot = True)
# print(segments)


seg = [[748, 752], [767, 768], [774, 777], [845, 851], [857, 867], [870, 1777], [1797, 1858], [1865, 1872], [1885, 2307], [2313, 2321], [2328, 2338], [2341, 2348], [2364, 2375], [2380, 2388], [2397, 2401], [2404, 2421], [2429, 2776], [2804, 2830], [2834, 2846], [2849, 2854], [2860, 2861], [2864, 2874], [2886, 2890], [2929, 2957], [2962, 2989], [2992, 2998], [3047, 3054], [3058, 3063], [3160, 3168], [3173, 3174], [3177, 3181], [3186, 3190], [3199, 3203], [3206, 3210], [3223, 3284], [3287, 3595]]

#seg = [[747.5, 752.0], [846.0, 851.0], [856.5, 867.5], [870.5, 1770.0], [1772.0, 1777.5], [1797.0, 1841.0], [1842.5, 1857.5], [1864.5, 1871.5], [1885.0, 2305.5], [2313.0, 2321.5], [2328.0, 2331.0], [2333.0, 2338.5], [2340.5, 2348.0], [2356.0, 2356.5], [2363.5, 2375.5], [2380.0, 2386.0], [2397.0, 2400.0], [2405.0, 2421.0], [2429.0, 2457.5], [2459.5, 2728.5], [2730.5, 2768.0], [2770.0, 2776.0], [2805.5, 2817.5], [2822.5, 2830.5], [2834.5, 2845.0], [2849.0, 2854.0], [2863.5, 2874.5], [2886.0, 2889.5], [2929.0, 2946.0], [2947.5, 2957.5], [2961.5, 2980.0], [2982.5, 2989.0], [2992.0, 2998.0], [3047.0, 3054.5], [3058.0, 3063.5], [3159.5, 3168.5], [3176.5, 3181.0], [3186.0, 3190.0], [3199.5, 3203.5], [3206.0, 3210.0], [3223.0, 3284.5], [3286.5, 3367.5], [3369.5, 3443.0], [3444.5, 3595.0]]


PATH = r'D:\birdsound'

from pydub import AudioSegment
# wav分割函数
def split_save_wav(filename, begin, end):
    sound = AudioSegment.from_wav(PATH+ '\\' +filename+".wav")
    duration = sound.duration_seconds * 1000  # 音频时长（ms）
    if 0 <= begin < end <= duration is False:
        print("split_save_wav - errors in begin and end")
        return
    cut_wav = sound[begin:end]  # 以毫秒为单位截取[begin, end]区间的音频
    savefilename = filename \
                   +"-"+str(begin)\
                   +"-"+str(end)
    cut_wav.export(r'D:\birdsound\divide2(5-1)' + '\\' + savefilename + '.wav', format='wav')  # 存储新的wav文件
    print("已导出到"+r'D:\birdsound\divide2(5-1)' + '\\' + savefilename + '.wav')

for i in seg:
    split_save_wav("after", int(i[0]*1000), int(i[1]*1000))