빌드 및 설치
============

빌드 환경
---------

### 컴파일러
khaiii는 C++14로 개발했으므로 이를 지원하는 컴파일러가 필요합니다.  
리눅스의 gcc 혹은 맥 OS의 clang의 경우 아래 환경에서 테스트했습니다.

OS 및 버전   | 컴파일러 버전
-------------|--------------
CentOS 7.2   | gcc-5.3.1 (devtoolset-4)
Ubuntu 16.04 | gcc-5.4.0
MacOS Mojave | LLVM-10.0.0


### 빌드 툴
[CMake](https://cmake.org/) 3.10 이상이 필요합니다.


### Python
khaiii는 실행하기 위한 프로그램과 함께 리소스(사전)를 필요로 합니다.  
이 리소스는 Python 스크립트를 통해 빌드되는데, 이때 Python 3.6 이상의 버전이 필요합니다.  
또한 몇 가지 패키지들을 필요로 하므로 pip를 통해 아래와 같이 설치합니다.

```bash
pip install -r requirements.txt
```

필요한 패키지는 다음과 같습니다.
* numpy
* torch==0.4.1
* tqdm

다른 패키지들은 특별히 버전을 명시하지 않았지만,  
[PyTorch](https://pytorch.org/)의 경우 저장된 모델 파일을 읽어 들이기 위해 0.4 버전이 꼭 필요합니다.



빌드
----

### 프로그램
build 디렉터리를 생성하고 아래와 같이 cmake를 실행합니다.

```bash
mkdir build
cd build
cmake ..
```

만약, Command Line 도구를 설치까지 하신다면,  
설치 경로를 바꾸기 위해서는 아래 명령대로 `PREFIX`를 지정하여 실행합니다.  
(기본값: /usr/local)
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
```

cmake를 실행하면 [Hunter](https://github.com/ruslo/hunter)를 통해  
필요한 패키지들의 소스코드를 GitHub으로부터 $HOME/.hunter 위치에 자동으로 내려받고 빌드합니다.

성공적으로 Makefile이 생성되면 다음과 같이 빌드합니다.

```bash
make all
```

성공적으로 빌드가 되면 build 디렉터리 아래에 다음과 같은 파일이 생성됩니다.

* bin: 디렉터리
  - khaiii: 실행 프로그램
* lib: 디렉터리
  - libkhaiii.so: shared 라이브러리 (맥 OS의 경우 libkhaiii.dylib)
  - libkhaiii.so.X
  - libkhaiii.so.X.Y
* test: 디렉터리
  - khaiii: 테스트 프로그램


### 리소스
bin 디렉터리 아래에 생성된 khaiii 프로그램을 실행하기 위해서는 리소스를 빌드해야 합니다.  
Makefile이 생성된 후 다음과 같이 빌드합니다.

```bash
make resource
```

성공적으로 빌드가 되면 share/khaiii 디렉터리 아래에 필요한 리소스 파일들이 빌드됩니다.

`make resource` 명령은 base 모델을 빌드합니다. large 모델을 빌드하고자 할 경우 다음과 같이 빌드하면 됩니다.

```bash
make large_resource
```


### 테스트
실행 프로그램과 리소스가 준비되면 build 디렉터리 아래에서 다음과 같이 실행해 봅니다.

```
$ ./bin/khaiii --rsc-dir=./share/khaiii
[2018-12-24 23:59:59.999] [Resource] [info] NN model loaded                   <== (로그)
[2018-12-24 23:59:59.999] [Preanal] [info] preanal dictionary opened          <== (로그)
[2018-12-24 23:59:59.999] [ErrPatch] [info] errpatch dictionary opened        <== (로그)
[2018-12-24 23:59:59.999] [Restore] [info] restore dictionary opened          <== (로그)
[2018-12-24 23:59:59.999] [Resource] [info] PoS tagger opened                 <== (로그)
안녕, 세상.        <== (입력)
안녕,	안녕/IC + ,/SP         <== (출력)
세상.	세상/NNG + ./SF        <== (출력)

```

다음과 같이 테스트 프로그램을 실행하면 정상 동작 여부를 알 수 있습니다.

```
$ ctest
Test project /home/khaiii/build
    Start 1: test_khaiii
1/1 Test #1: test_khaiii ......................   Passed    0.02 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.03 sec
```



설치
----

### 프로그램과 리소스
프로그램과 리소스가 정상적으로 빌드되었다면 다음 명령을 통해 설치할 수 있습니다.

```bash
make install
```

이렇게 하면 `PREFIX` 경로 아래에 bin, include, lib, share 디렉터리를 생성하게 됩니다.  
(`PREFIX` 경로 변경은 [빌드](#빌드) 참고)


### Python 바인딩
Python 바인딩은 일단 빌드가 성공적으로 되었다면 다음과 같이 pip 명령을 이용하여 설치할 수 있습니다.

```bash
make package_python
cd package_python
pip install .
```

정상적으로 설치되었다면 다음과 같이 Python 인터프리터에서 사용할 수 있습니다.

```python
>>> import khaiii
>>> api = khaiii.KhaiiiApi()
>>> api.open()
[2018-12-24 23:59:59.999] [Resource] [info] NN model loaded
[2018-12-24 23:59:59.999] [Preanal] [info] preanal dictionary opened
[2018-12-24 23:59:59.999] [ErrPatch] [info] errpatch dictionary opened
[2018-12-24 23:59:59.999] [Restore] [info] restore dictionary opened
[2018-12-24 23:59:59.999] [Resource] [info] PoS tagger opened
>>> for word in api.analyze('안녕, 세상.'):
...     morphs_str = ' + '.join([(m.lex + '/' + m.tag) for m in word.morphs])
...     print(f'{word.lex}\t{morphs_str}')
...
안녕,	안녕/IC + ,/SP
세상.	세상/NNG + ./SF
```
