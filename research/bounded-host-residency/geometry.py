# expert geometry of a GGUF: bank bytes, per-expert bytes, offsets of each expert's parts
import struct,sys,re,json
def load(path):
    f=open(path,'rb')
    def rd(fmt): return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
    def rs(): n=rd('Q'); return f.read(n).decode('utf-8','replace')
    SZ={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def rv(t):
        if t==8: return rs()
        if t==9:
            at=rd('I'); n=rd('Q'); a=[rv(at) for _ in range(n)]; return a if n<64 else ('arr',n)
        return rd(SZ[t])
    assert f.read(4)==b'GGUF'; rd('I'); nt=rd('Q'); nk=rd('Q')
    kv={}
    for _ in range(nk):
        k=rs(); t=rd('I'); kv[k]=rv(t)
    align=kv.get('general.alignment',32)
    ts=[]
    for _ in range(nt):
        n=rs(); nd=rd('I'); sh=[rd('Q') for _ in range(nd)]; ty=rd('I'); off=rd('Q'); ts.append([n,sh,ty,off])
    data0=(f.tell()+align-1)//align*align
    import os; end=os.path.getsize(path)
    s=sorted(ts,key=lambda x:x[3])
    for i,t in enumerate(s): t.append((s[i+1][3] if i+1<len(s) else end-data0)-t[3])
    return kv,ts,data0,end
if __name__=='__main__':
    kv,ts,d0,end=load(sys.argv[1])
    arch=kv['general.architecture']
    E=kv.get(arch+'.expert_count'); K=kv.get(arch+'.expert_used_count')
    bank=[t for t in ts if re.search(r'ffn_(up|down|gate|gate_up)_exps\.weight$',t[0])]
    B=sum(t[4] for t in bank); tot=end
    layers=sorted({int(t[0].split('.')[1]) for t in bank})
    per_expert=B/len(layers)/E
    parts=sorted({re.search(r'ffn_(\w+)_exps',t[0]).group(1) for t in bank})
    # layout of one layer's banks
    l0=[t for t in bank if t[0].startswith(f'blk.{layers[0]}.')]
    l0s=sorted(l0,key=lambda x:x[3])
    gaps=[(l0s[i+1][3]-(l0s[i][3]+l0s[i][4])) for i in range(len(l0s)-1)]
    print(json.dumps({'model':sys.argv[1].split('/')[-1],'arch':arch,'n_expert':E,'n_used':K,'moe_layers':len(layers),
        'file_gib':round(tot/2**30,3),'bank_gib':round(B/2**30,3),'nonbank_gib':round((tot-B)/2**30,3),
        'per_expert_mib':round(per_expert/2**20,3),'parts':parts,
        'part_bytes_per_expert_mib':{re.search(r'ffn_(\w+)_exps',t[0]).group(1):round(t[4]/E/2**20,3) for t in l0},
        'layer0_part_offsets_gib':[round((d0+t[3])/2**30,4) for t in l0s],'layer0_gaps_between_parts_mib':[round(g/2**20,2) for g in gaps]}))
