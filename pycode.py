cw_trans = {}
tech = ["benign", "AE", "Down-Up", "LPC", "Quant", "ANR", "SNR", "Ours"]
for te in tech:
    cw_trans[te] = []
with open(file) as f:
    lines = f.readlines()
    for line in lines:  # line: 
        types = line.split("-")[1] "1-benign.wav"
        trans = line.split(":")[-1]
        cw_trans[ele].append(trans)