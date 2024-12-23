import argparse
import yaml
from lib.aptm.models.model_retrieval import APTM_Retrieval
from lib.aptm.models.tokenization_bert import BertTokenizer
from torchvision import transforms
from PIL import Image

from lib.aptm.reTools import inference_attr, inference_text, train_text


class APTM:
    def __init__(self, config, task, checkpoint, device = 'cuda'):
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader) # Reading the config file
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(self.config['text_encoder'])
        self.model = APTM_Retrieval(config=self.config)
        if self.config['load_pretrained']:
            self.model.load_pretrained(checkpoint, self.config, is_eval=True)
        self.model = self.model.to(self.device)
        
        # Freeze all model parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # cuhk
        cuhk_norm = transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555))
        # icfg
        icfg_norm = transforms.Normalize((0.30941582, 0.28956893, 0.30347288), (0.25849792, 0.24547698, 0.2366199))
        # pa100k
        pa100k_norm = transforms.Normalize((0.46485138, 0.45038012, 0.4632019), (0.25088054, 0.24609283, 0.24240193))
        # rstp
        rstp_norm = transforms.Normalize((0.27722597, 0.26065794, 0.3036557), (0.2609547, 0.2508087, 0.25293276))
        # gene
        gene_norm = transforms.Normalize((0.4416847, 0.41812873, 0.4237452), (0.3088255, 0.29743394, 0.301009))

        if task == 'cuhk':
            train_norm = cuhk_norm
            test_norm = cuhk_norm
        elif task == 'icfg':
            train_norm = icfg_norm
            test_norm = icfg_norm
        elif task == 'pa100k':
            train_norm = pa100k_norm
            test_norm = pa100k_norm
        elif task == 'rstp':
            train_norm = rstp_norm
            test_norm = rstp_norm
        elif task == 'gene':
            train_norm = gene_norm
            test_norm = gene_norm
        else:
            assert("error")

        self.image_transform = transforms.Compose([
            transforms.Resize((self.config['h'], self.config['w']), interpolation=3), # 3 for InterpolationMode.BICUBIC
            transforms.ToTensor(),
            test_norm,
        ])

    def inference_calculate_score(self, text, images, CNN_image_features, CNN_image_alpha):
        # single_text, many_images
        # text: list
        # images: list

        tacked_images = []
        for i, image in enumerate(images):
            image = self.image_transform(image)
            tacked_images.append(image)

        # Bigger is better
        score_test_i2t_attr = inference_attr(self.model, text, tacked_images,
                                                    self.tokenizer, self.device, self.config)

        # Bigger is better
        score_test_i2t_text = inference_text(self.model, text, tacked_images, CNN_image_features, CNN_image_alpha,
                                                    self.tokenizer, self.device, self.config)
            
        return score_test_i2t_attr[0], score_test_i2t_text[0] # Since only one line of text is input each time, [0] is used.

    def train(self, texts, images, CNN_image_features, CNN_image_alpha):
        # many_texts, many_images
        # texts: list
        # images: list

        tacked_images = []
        for i, image in enumerate(images):
            image = self.image_transform(image)
            tacked_images.append(image)

        # Bigger is better
        # range:[0, 1]
        score_train_i2t_text = train_text(self.model, texts, tacked_images, CNN_image_features, CNN_image_alpha,
                                                    self.tokenizer, self.device, self.config)
        
        
        # # Check whether the parameters are frozen:
        # print("#######################################################################################################################")
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

        return score_train_i2t_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()

    aptm_inference = APTM(args.config, args.task, args.checkpoint)

    text = [
        # Format 1：
        # 'a man wearing a yellow coat',
        # 'a man wearing a black coat and black trousers',
        # 'a man wearing a white coat',
        # 'a man wearing a gray coat',
        # 'a man wearing a blue helmet, blue coat and blue trousers',
        # 'a man wearing an orange coat, black trousers and white shoes',
        # 'a man wearing a white coat, black trousers and black shoes',
        # 'a man wearing a white coat, black trousers and white shoes',
        # 'a man in a white coat and black trousers',
        # 'a man in a black coat, riding an electric bike',
        # 'a man wearing black coat and black trousers',
        # 'a man wearing a yellow coat and black trousers',
        # 'a man wearing a pink coat',
        # Format 2：
        # 'A man wearing a yellow coat.',
        # 'A man wearing a black coat and black trousers.',
        # 'A man wearing a white coat.',
        'A man wearing a pink coat.',
        # 'A man wearing a gray coat.',
        # 'A man wearing a blue helmet, blue coat and blue trousers.',
        # 'A man wearing an orange coat, black trousers and white shoes.',
        # 'A man wearing a white coat, black trousers and black shoes.',
        # 'A man wearing a white coat, black trousers and white shoes.',
        # 'A man in a white coat and black trousers.',
        # 'A man in a black coat, riding an electric bike.',
        # 'A man wearing black coat and black trousers.',
        # 'A man wearing a yellow coat and black trousers.',
        # 'A man wearing a pink coat.',
            ]

    images = []

    for i in range(20):
        image_path = "/mnt/A/hust_csj/Code/APTM/test/images/" + str(i+1) + ".png"
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    score_test_i2t_attr, score_test_i2t_text = aptm_inference.inference_calculate_score(text, images)

    print("Enter text:",text[0])
    print(score_test_i2t_attr)
    print(score_test_i2t_text)
