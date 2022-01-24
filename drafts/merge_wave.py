import wave

# infiles = [r"D:\birdsound\test211112-000000-000100.wav",
#            r"D:\birdsound\test211112-002300-002400.wav"]
# outfile = r"D:\birdsound\merge.wav"

infiles = [r"D:\birdsound\after_denoising1.wav",
           r"D:\birdsound\after_denoising1.wav"]
outfile = r"D:\birdsound\merge2.wav"


data= []
for infile in infiles:
    w = wave.open(infile, 'rb')
    data.append( [w.getparams(), w.readframes(w.getnframes())] )
    w.close()

output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
output.writeframes(data[0][1])
output.writeframes(data[1][1])
output.close()
