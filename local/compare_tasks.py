import json, statistics, sys
def load(p):
    d=json.load(open(p)); x=d['targets']['v100-flash-medium']; return {r['id']:r for r in x['rows']}, x['wall_s']
a,aw=load(sys.argv[1]); b,bw=load(sys.argv[2])
ids=sorted(a)
pa=sum(a[i]['status']=='pass' for i in ids); pb=sum(b[i]['status']=='pass' for i in ids)
print(f"passes A {pa}/{len(ids)}  B {pb}/{len(ids)}   suite wall A {aw:.0f}s B {bw:.0f}s")
print(f"median task wall A {statistics.median(a[i]['wall_s'] for i in ids):.1f}s  B {statistics.median(b[i]['wall_s'] for i in ids):.1f}s")
print(f"median decode A {statistics.median(a[i]['decode_ts'] for i in ids):.1f}  B {statistics.median(b[i]['decode_ts'] for i in ids):.1f} tok/s; median gen tokens A {statistics.median(a[i]['gen_n'] for i in ids):.0f} B {statistics.median(b[i]['gen_n'] for i in ids):.0f}")
print("tasks whose status changed:")
for i in ids:
    if a[i]['status']!=b[i]['status'] or a[i]['ok']!=b[i]['ok']: print(f"  {i}: A {a[i]['status']} {a[i]['ok']}/{a[i]['total']}  B {b[i]['status']} {b[i]['ok']}/{b[i]['total']}  ({a[i]['detail'][:50]!r} -> {b[i]['detail'][:50]!r})")
