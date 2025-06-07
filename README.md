# training


### WSL network setup

In windows powershell, create a file under home directory, named .wslconfig, 
with this content:

```
[wsl2]
generateResolvConf=false
networkingMode=mirrored
dnsTunneling=true
```

Go to windows settings, Network & Internet->Proxy, check the "Don't use the proxy server for local(intranet) addresses" box.

### WSL pyenv installation
see https://koki-nakamura22.github.io/blog/posts/python/installing-pyenv-pyenv-on-wsl-ubuntu1804/

