.PHONY: all

all:  linux_requirements python_requirements MFA_thirdparty MFA_pretrained

linux_requirements:
	sudo apt-get update && sudo apt-get install -y libsndfile1 libopenblas-dev

python_requirements:
	conda env update -f environment.yaml
	pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
	pip isntall torchaudio==0.9.0

MFA_thirdparty:

	mfa thirdparty download
	mfa thirdparty validate

MFA_pretrained:
	mfa download acoustic english
	mfa download g2p english_g2p
	mfa download dictionary english
