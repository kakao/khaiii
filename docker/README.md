# khaiii on Docker

Docker 컨테이너에서 `khaiii`를 사용할 수 있습니다.

## Using Public Image

(TBD)

## Building Image

소스를 변경하거나, 모델을 수정한 경우 이미지를 직접 빌드해서 사용할 수 있습니다.

### [continuumio/anaconda3](./for-continuumio-anaconda3-image/Dockerfile)

`continuumio/anaconda3` 이미지에 `khaiii` Python 패키지가 추가된 이미지입니다.

### pytorch

- [0.4.1](./pytorch-0.4.1/Dockerfile)
- [1.1.0](./pytorch-1.1.0/Dockerfile)

Pytorch 이미지에서 빌드된 `khaiii` 바이너리와 모델이 포함된 이미지입니다.

작업환경의 소스를 기준으로 이미지를 빌드하기 때문에 `build context`를 프로젝트의 `root` 디렉터리로 명시해주셔야 합니다.

```sh
# /docker 디렉터리 내부에서 pytorch-1.1.0 기반의 이미지를 빌드하는 경우
docker build .. -f ./pytorch-1.1.0/Dockerfile -t khaiii:my-custom-model
```

바이너리 이외의 산출물(ex. 라이브러리)이 필요한 경우 다음과 같이 `--target` 옵션을 통해 빌드 환경의 이미지를 빌드할 수 있습니다.

```sh
# /docker 디렉터리 내부에서 pytorch-1.1.0 기반의 빌드 환경 이미지를 빌드하는 경우
docker build .. -f ./pytorch-1.1.0/Dockerfile -t khaiii:my-custom-build --target builder
```
