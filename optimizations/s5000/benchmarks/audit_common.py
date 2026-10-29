from pathlib import Path
import os,hashlib
def paths():
 out=[]
 for line in Path("/proc/self/maps").read_text().splitlines():
  if "/libmudnn" not in line:continue
  assert "(deleted)" not in line,line
  p=line.split()[-1]
  if p.startswith("/"):out.append(str(Path(p).resolve()))
 return sorted(set(out))
def checked_paths():
 out=paths();assert out,"muDNN not loaded"
 root=Path(os.environ["MUDNN_EXPECT_ROOT"]).resolve()
 assert all(Path(p).is_relative_to(root) for p in out),(str(root),out)
 return out
def identities():
 out=[]
 for n in checked_paths():
  p=Path(n);h=hashlib.sha256()
  with p.open("rb") as f:
   for chunk in iter(lambda:f.read(16*1024**2),b""):h.update(chunk)
  out.append(dict(path=n,bytes=p.stat().st_size,sha256=h.hexdigest()))
 return out
