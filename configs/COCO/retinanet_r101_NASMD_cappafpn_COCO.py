_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet_MD',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
	mdconv=dict(type='NASConv', use_deform=False),
        stage_with_mdconv=(False, True, True, True),
        style='pytorch'),
    neck=dict(
        type='CAPS_PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
        num_pri_caps=12,
        num_att_caps=1,
        update_routing_pri=True,
        num_att_routing=5),
    bbox_head=dict(
		num_classes=80
	))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
