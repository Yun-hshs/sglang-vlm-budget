"""Transparent lightweight metrics; official dataset evaluators remain authoritative."""
import re
import string
import unicodedata

def normalize(text):
    text=unicodedata.normalize("NFKC",str(text)).lower().strip()
    text=re.sub(r"(?<=\d),(?=\d)","",text)
    text=re.sub(r"[^\w\s.]"," ",text)
    text=re.sub(r"\b(a|an|the)\b"," ",text)
    numbers=dict(zip("zero one two three four five six seven eight nine ten".split(),map(str,range(11))))
    return " ".join(numbers.get(t,t) for t in text.split()).strip(".")

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

def score(prediction, answers, metric="em"):
    if not answers: raise ValueError("Reference answers required")
    p=normalize(prediction); refs=[normalize(a) for a in answers]
    if metric=="em": return float(p in refs)
    if metric=="vqa":
        if len(refs)!=10: raise ValueError("VQA consensus requires 10 annotations")
        return sum(min(1,sum(p==r for j,r in enumerate(refs) if i!=j)/3) for i in range(10))/10
    if metric=="anls":
        values=[1-edit_distance(p,r)/max(len(p),len(r),1) for r in refs]
        best=max(values); return best if best>0.5 else 0.0
    raise ValueError("Unknown metric")
