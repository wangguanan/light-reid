import os
import torch
import lightreid
from lightreid.engine import Engine
from lightreid.utils import MultiItemAverageMeter, Logging


class AGWEngine(Engine):

    def __init__(self, results_dir, datamanager, model, criterion, optimizer, use_gpu, eval_metric='cosine',
                 light_model=False, light_feat=False, light_search=False,
                 kl_t=4.0, circle_s=None, circle_m=None):

        # base settings
        self.results_dir = os.path.join(results_dir,
                           'lightmodel(klt{}-s{}-m{})-lightfeat({})-lightsearch({})'.\
                               format(light_model,  circle_s, circle_m, light_feat, light_search))
        self.datamanager = datamanager
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.eval_metric = eval_metric

        self.loss_meter = MultiItemAverageMeter()
        os.makedirs(self.results_dir, exist_ok=True)
        self.logging = Logging(os.path.join(self.results_dir, 'logging.txt'))

        # optinal settings for light-reid learning
        self.light_model = light_model
        self.light_feat = light_feat
        self.light_search = light_search
        self.logging('\n' + '****'*5 + ' light-reid settings ' + '****'*5)
        self.logging('light_model:  {}'.format(light_model))
        self.logging('light_feat:  {}'.format(light_feat))
        self.logging('light_search:  {}'.format(light_search))
        self.logging('****'*5 + ' light-reid settings ' + '****'*5 + '\n')

        # if enable light_model, learn small model with distillation
        # update model to a small model (res18)
        # load teacher model from model_teacher (should be trained before)
        # add KLLoss(distillation loss) to criterion
        if self.light_model:
            # load teacher model
            teacher_path = os.path.join(results_dir, 'lightmodel(False)-lightfeat(False)-lightsearch(False)/final_model.pth.tar')
            assert os.path.exists(teacher_path), \
                'lightmodel was enabled, expect {} as a teachder but file not exists'.format(self.model_teachder)
            model_t = torch.load(teacher_path)
            self.model_t = model_t.to(self.device).eval()
            self.logging('[light_model was enabled] load teacher model from {}'.format(teacher_path))
            # modify model to a small model (ResNet18 as default here)
            pretrained, last_stride_one = self.model.backbone.pretrained, self.model.backbone.last_stride_one
            self.model.backbone.__init__('resnet18', pretrained, last_stride_one)
            self.model.head.__init__(self.model.backbone.dim, self.model.head.class_num, self.model.head.classifier)
            if self.model.head.classifier.__class__.__name__ == 'Circle':
                circle_s = self.model.head.classifier._s if circle_s is None else circle_s
                circle_m = self.model.head.classifier._m if circle_m is None else circle_m
                self.model.head.classifier.__init__(
                    self.model.backbone.dim, self.model.head.classifier._num_classes, scale=circle_s, margin=circle_m)
                self.logging('[light_model was enabled] Classifier layer was set as Circle with scale {} amd margin {}'.format(circle_s, circle_m))
            self.logging('[light_model was enabled] modify model to ResNet18')
            print(self.model)
            # update optimizer
            optimizer_defaults = self.optimizer.optimizer.defaults
            self.optimizer.optimizer.__init__(self.model.parameters(), **optimizer_defaults)
            self.logging('[light_model was enabled] update optimizer parameters')
            # add KLLoss to criterion
            self.criterion.criterion_list.append(
                {'criterion': lightreid.losses.KLLoss(t=kl_t), 'weight': 1.0})
            self.logging('[light_model was enabled] add KLLoss for model distillation with temperature {}'.format(kl_t))

        # if enable light_feat,
        # learn binary codes NOT real-value features
        # evaluate with hamming metric, NOT cosine NEITHER euclidean metrics
        if self.light_feat:
            self.model.enable_tanh()
            self.eval_metric = 'hamming'
            self.logging('[light_feat was enabled] model learn binary codes, and is evluated with hamming distance')
            self.logging('[light_feat was enabled] update eval_metric from {} to hamming by setting self.eval_metric=hamming'.format(eval_metric))


        # if enable light_search,
        # learn binary codes of multiple length with pyramid-head
        # and search with coarse2fine strategy
        if self.light_search:
            # modify head to pyramid-head
            in_dim, class_num = self.model.head.in_dim, self.model.head.class_num
            self.model.head = lightreid.models.CodePyramid(
                in_dim=in_dim, out_dims=[2048, 512, 128, 32], class_num=class_num)
            self.logging('[light_search was enabled] learn multiple codes with {}'.format(self.model.head.__class__.__name__))
            # update optimizer parameters
            optimizer_defaults = self.optimizer.optimizer.defaults
            self.optimizer.optimizer.__init__(self.model.parameters(), **optimizer_defaults)
            self.logging('[light_search was enabled] update optimizer parameters')
            # add self-ditillation loss loss for pyramid-haed
            self.criterion.criterion_list.extend([
                {'criterion': lightreid.losses.ProbSelfDistillLoss(), 'weight': 1.0},
                {'criterion': lightreid.losses.SIMSelfDistillLoss(), 'weight': 1000.0},
            ])
            self.logging('[light_search was enabled] add ProbSelfDistillLoss and SIMSelfDistillLoss')

        self.model = self.model.to(self.device)