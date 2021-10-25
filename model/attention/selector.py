from .width_att import WAM

def get_att_layer(att_name):
    return dict(
        WAM = WAM
        # You can add more attention modules to select
    )[att_name]

