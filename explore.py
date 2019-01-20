from PIL import Image
import config as configlib
import model as modellib
import data as datalib
import tools.canvas as convaslib


def load_model():
    """Loading Mask R-CNN for coco."""
    config = configlib.CocoConfig()
    net = modellib.MaskRCNN(config=config, model_dir="logs")
    net.load_weights("models/mask_rcnn_coco.pth")
    net.eval()

    net.training = False
    net = net.cuda()

    return net

config = configlib.Config()
model = load_model()
image_file = "images/car58a54312d.jpg"


def show_fpn():
    """Ouput fpn network."""
    print(model.fpn)

show_fpn()


def show_image():
    """Show network input image."""
    image = Image.open(image_file)
    image, scale, cropbox = datalib.encode_image(image, 800, 1024)
    image.show()

show_image()


def show_p2():
    """Show P2."""
    image = Image.open(image_file)
    image, scale, cropbox = datalib.encode_image(image, 800, 1024)
    normalized_image = datalib.normalize_image(image, config.MEAN_PIXEL)
    normalized_image.unsqueeze_(0)
    normalized_image = normalized_image.cuda()
    [p2, p3, p4, p5, p6] = model.fpn(normalized_image)
    some_of_p2 = p2[:, :16, :, :]
    convaslib.tensor_show(some_of_p2)

show_p2()


def draw_anchors():
    """Draw anchors base on P3 dimension."""
    import data
    import utils
    image = Image.open("images/car58a54312d.jpg")
    image, scale, crop = data.encode_image(image, 800, 1024)
    anchors = utils.create_anchors(scales=64, ratios=[0.5, 1, 2], shape=[128, 128], feature_stride=8, anchor_stride=1)
    data.draw_anchors(image, anchors)

draw_anchors()


def refine_rpn():
    """Refine rpn."""
    image = Image.open(image_file)
    image, scale, cropbox = datalib.encode_image(image, 800, 1024)

    normalized_image = datalib.normalize_image(image, config.MEAN_PIXEL)
    normalized_image.unsqueeze_(0)
    normalized_image = normalized_image.cuda()
    [p2, p3, p4, p5, p6] = model.fpn(normalized_image)

    rpn_class_logits, rpn_class, rpn_bbox = model.rpn_detect([p2, p3, p4, p5, p6])
    rpn_rois = model.rpn_refine(rpn_class, rpn_bbox)
    rpn_rois.squeeze_(0)

    boxes = datalib.boxes_scale(rpn_rois, [1024, 1024, 1024, 1024])
    datalib.blend_image(image, None, boxes, masks=None, scores=None).show()

refine_rpn()


def detection():
    """Detection boxes."""
    image = Image.open(image_file)
    image, scale, cropbox = datalib.encode_image(image, 800, 1024)

    normalized_image = datalib.normalize_image(image, config.MEAN_PIXEL)
    normalized_image.unsqueeze_(0)
    normalized_image = normalized_image.cuda()
    [p2, p3, p4, p5, p6] = model.fpn(normalized_image)

    rpn_class_logits, rpn_class, rpn_bbox = model.rpn_detect([p2, p3, p4, p5, p6])
    rpn_rois = model.rpn_refine(rpn_class, rpn_bbox)
    rpn_rois.squeeze_(0)

    # Detections
    mrn_class_logits, mrn_class, mrn_bbox = model.mrn_detect([p2, p3, p4, p5], rpn_rois)
    mrn_class_ids, mrn_scores, mrn_boxes = model.mrn_refine(rpn_rois, mrn_class, mrn_bbox, [0, 0, 1024, 1024])
    boxes = mrn_boxes.squeeze(0).detach().cpu()
    datalib.blend_image(image, None, boxes, masks=None, scores=None).show()


detection()
