import torch
import core.metrics as Metrics

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

def get_cand_err2(model, cand, data, args):
    avg_psnr = 0.0
    idx = 0
    for _,  val_data in enumerate(data):
        idx += 1
        model.feed_data(val_data)
        model.test(cand=cand, continous=True)

        visuals = model.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        sr_img = Metrics.tensor2img(visuals['SR'][-1])
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        avg_psnr += psnr
    avg_psnr = avg_psnr / idx
    return avg_psnr


