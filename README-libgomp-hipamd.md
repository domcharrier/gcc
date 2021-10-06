Either clone the whole repository or checkout the relevant files via git `sparse-checkout`.

```bash
#!/usr/bin/env bash
mkdir GCC
cd GCC/
git init
git sparse-checkout init
git remote add -f origin https://github.com/gcc-mirror/gcc
git fetch
git sparse-checkout set libgomp config ltoptions.m4 ltsugar.m4 ltversion.m4 lt~obsolete.m4 libtool.m4 ltgcc.m4 ltmain.sh multilib.am config-ml.in
```

Developer: Generate configure script & Makefile via autotools

```
cd libgomp/
aclocal -I ../config
autoconf -f
automake --add-missing
```

User: Configure and build libgomp with `hipamd` backend:

```
cd libgomp/
./configure 
```
