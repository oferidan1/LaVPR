import torch

from vpr_models.resizing_wrapper import ResizingWrapper

from vpr_models import clique_mining, convap, mixvpr, netvlad, sfrs, boq, dinomix, cricavpr_small



def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "sfrs":
        model = sfrs.SFRSModel()
    elif method == "netvlad":
        model = netvlad.NetVLAD(descriptors_dimension=descriptors_dimension)
    elif method == "cosplace":
        model = torch.hub.load(
            "gmberton/cosplace", "get_trained_model", backbone=backbone, fc_output_dim=descriptors_dimension
        )
    elif method == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=descriptors_dimension)
    elif method == "convap":
        model = convap.get_convap(descriptors_dimension=descriptors_dimension)
    elif method == "eigenplaces":
        model = torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model", backbone=backbone, fc_output_dim=descriptors_dimension
        )
    elif method == "eigenplaces-indoor":
        model = torch.hub.load(
            "Enrico-Chiavassa/Indoor-VPR", "get_trained_model", backbone=backbone, fc_output_dim=descriptors_dimension
        )
    elif method.startswith("anyloc"):
        domain = method.split("-")[1]
        anyloc = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", domain=domain, device="cuda")
        model = ResizingWrapper(anyloc, resize_type="dino_v2_resize")
    elif method == "salad":
        salad = torch.hub.load("serizba/salad", "dinov2_salad")
        model = ResizingWrapper(salad, resize_type="dino_v2_resize")
    elif method == "clique-mining":
        clique_mining_model = clique_mining.get_clique_mining_model()
        model = ResizingWrapper(clique_mining_model, resize_type="dino_v2_resize")
    elif method == "salad-indoor":
        salad_indoor = torch.hub.load(
            "Enrico-Chiavassa/Indoor-VPR", "get_trained_model", method="salad", backbone="Dinov2", fc_output_dim=8448
        )
        model = ResizingWrapper(salad_indoor, resize_type="dino_v2_resize")
    elif method == "cricavpr":
        cricavpr = torch.hub.load("Lu-Feng/CricaVPR", "trained_model")
        model = ResizingWrapper(cricavpr, resize_type=224)
    elif method == "cricavpr_small":
        cricavpr = cricavpr_small.get_cricavpr_small()
        model = ResizingWrapper(cricavpr, resize_type=224)    
    elif method == "megaloc":
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")

    elif method == "boq":
        model = boq.get_boq(backbone=backbone, descriptors_dimension=descriptors_dimension)

    elif method == "dinomix":
        model = dinomix.get_dino_mix()
        
    elif method == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        #model = ResizingWrapper(model, resize_type=224)

    return model
