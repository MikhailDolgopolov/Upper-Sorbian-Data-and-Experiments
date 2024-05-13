import pandas as pd

f = open("Data/clean_wikipedia/hsb_wikipedia_2016_30K-clean_words.txt","r", encoding="utf-8").read().splitlines()
out= 'Data/clean_wikipedia/manually_clean.txt'
file_to_delete = open(out,'w')
file_to_delete.close()
lines=[]
for line in f:
    a = line.split()
    if len(a) > 3:
        line = f"{a[0]}\t{'_'.join(a[1:-1])}\t{a[-1]}"
    lines.append(line+"\n")
with open(out, 'a', encoding="utf-8") as the_file:
    the_file.write("BadIndex\tWord\tFreq\n")
    the_file.writelines(lines[29:])
data = pd.read_csv("/Data/clean_wikipedia/manually_clean.txt",
                   delimiter="\t", encoding="utf-8", engine='python').drop("BadIndex", axis=1)
print(data)