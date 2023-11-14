import dataset
import models.crnn as crnn
import torch
import utils
from PIL import Image
from torch.autograd import Variable

from definitions import ROOT_DIR

model_path = ROOT_DIR + "/scr/text_recognition/crnn_pytorch/data/crnn.pth"
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def recognize_text(img_path: str) -> str:
    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert("L")
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(
        preds.data, preds_size.data, raw=False
    )  # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred
