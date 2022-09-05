import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace

# gets role for executing training job
iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName='{IAM_ROLE_WITH_SAGEMAKER_PERMISSIONS}')['Role']['Arn']
hyperparameters = {
	'model_name_or_path': 't5-small',
	'output_dir': '/opt/ml/model'
	# add your remaining hyperparameters
	# more info here https://github.com/huggingface/transformers/tree/v4.6.1/examples/pytorch/seq2seq
}

# git configuration to download our fine-tuning script
git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.6.1'}

# creates Hugging Face estimator
huggingface_estimator = HuggingFace(
	entry_point='run_translation.py',
	source_dir='./examples/pytorch/seq2seq',
	instance_type='ml.p3.2xlarge',
	instance_count=1,
	role=role,
	git_config=git_config,
	transformers_version='4.6.1',
	pytorch_version='1.7.1',
	py_version='py36',
	hyperparameters = hyperparameters
)

# starting the train job
huggingface_estimator.fit()