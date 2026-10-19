import csv, json, pathlib
A=pathlib.Path('/home/aim/Documents/Obsidian/_attachments/strata-v100-tuning-20261018')
soak=[r for r in csv.DictReader(open(A/'soak_135w.csv'))]
trace=[[round(float(r['t_s'])),round(float(r['gpu_w']))] for i,r in enumerate(soak) if i%3==0]
tasks=[r for r in csv.DictReader(open(A/'tasks_ab.csv'))]
tasks_js=[{'task':r['task'],'a':float(r['wall_oct13_s']),'b':float(r['wall_final_s']),'pa':int(r['pass_oct13']),'pb':int(r['pass_final'])} for r in tasks]
pk=max(soak,key=lambda r: float(r['gpu_w'])); data={'trace':trace,'tasks':tasks_js,'peak':[round(float(pk['t_s'])),round(float(pk['gpu_w']))]}
tpl=pathlib.Path(__file__).with_name('template.html').read_text()
out=tpl.replace('__DATA__', json.dumps(data, separators=(',',':')))
pathlib.Path(__file__).with_name('v100-report.html').write_text(out)
print('built', len(out), 'bytes; trace points', len(trace))
