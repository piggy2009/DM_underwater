import torch

def load_part_of_model(new_model, src_model_path, s):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict, strict=s)
    return new_model

def load_part_of_model2(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        k2 = k.replace('denoise_fn.', '')
        if k2 in m_dict.keys():
            # print(k)
            param = src_model.get(k)
            if param.shape == m_dict[k2].data.shape:
                m_dict[k2].data = param
                print('loading:', k)
            # else:
            #     print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict)
    return new_model