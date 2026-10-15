import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
print('pages:', len(r.pages))
out = open(sys.argv[2], 'w', encoding='utf-8')
for i, p in enumerate(r.pages):
    out.write('\n\n===== PAGE %d =====\n' % (i+1))
    try:
        out.write(p.extract_text() or '')
    except Exception as e:
        out.write('[err %s]' % e)
out.close()
