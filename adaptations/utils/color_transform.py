import math

import torch


@torch.no_grad()
class RGB_LAB_Converter:
    def __init__(self, device):
        # constant conversion matrices between color spaces: https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
        self._rgb2xyz = torch.tensor([[0.412453, 0.357580, 0.180423],
                                      [0.212671, 0.715160, 0.072169],
                                      [0.019334, 0.119193, 0.950227]]).to(device)
        self._rgb2xyz.requires_grad = False

        self._xyz2rgb = torch.linalg.inv(self._rgb2xyz)
        self._xyz2rgb.requires_grad = False

        self._white = torch.tensor([0.95047, 1., 1.08883]).to(device)
        self._white.requires_grad = False

        self.eps = 1e-8

    @torch.no_grad()
    def __call__(self, images_source_batch, images_target_batch):
        B, _, _, _ = images_source_batch.shape
        images_source_converted_list = []
        for i in range(B):
            images_source = images_source_batch[i]
            images_target = images_target_batch[i]
            torch.clamp(images_source, 0, 1)
            torch.clamp(images_target, 0, 1)

            images_source_lab = self._rgb2lab(images_source)
            mean_s = torch.mean(images_source_lab, dim=(1, 2), keepdim=True)
            # + self.eps to prevent NaN
            std_s = torch.std(images_source_lab, dim=(1, 2), keepdim=True) + self.eps

            images_target_lab = self._rgb2lab(images_target)
            mean_t = torch.mean(images_target_lab, dim=(1, 2), keepdim=True)
            # + self.eps to prevent NaN
            std_t = torch.std(images_target_lab, dim=(1, 2), keepdim=True) + self.eps

            images_source_lab_converted = (images_source_lab - mean_s) * std_t / std_s + mean_t

            images_source_converted = self._lab2rgb(images_source_lab_converted)
            images_source_converted_list.append(images_source_converted)
        images_source2target = torch.stack(images_source_converted_list, dim=0)
        return images_source2target

    @torch.no_grad()
    def _rgb2lab(self, rgb):
        arr = rgb.clone().type(torch.float32)

        # convert rgb -> xyz color domain
        mask = arr > 0.04045
        not_mask = torch.logical_not(mask)
        arr.masked_scatter_(mask, torch.pow((torch.masked_select(arr, mask) + 0.055) / 1.055, 2.4))
        arr.masked_scatter_(not_mask, torch.masked_select(arr, not_mask) / 12.92)

        xyz = torch.tensordot(torch.t(self._rgb2xyz), arr, dims=([0], [0]))

        # scale by CIE XYZ tristimulus values of the reference white point
        arr = torch.mul(xyz, 1 / self._white.type(xyz.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1))

        # nonlinear distortion and linear transformation
        mask = arr > 0.008856
        not_mask = torch.logical_not(mask)
        arr.masked_scatter_(mask, torch.pow(torch.masked_select(arr, mask), 1 / 3))
        arr.masked_scatter_(not_mask, 7.787 * torch.masked_select(arr, not_mask) + 16 / 166)

        # get each channel as individual tensors
        x, y, z = arr[0], arr[1], arr[2]

        # vector scaling
        L = (116. * y) - 16.
        a = 500.0 * (x - y)
        b = 200.0 * (y - z)

        # OpenCV format
        L *= 2.55
        a += 128
        b += 128

        # finally, get LAB color domain
        return torch.stack([L, a, b], dim=0)

    @torch.no_grad()
    def _lab2rgb(self, lab):
        lab = lab.clone().type(torch.float32)

        # rescale back from OpenCV format and extract LAB channel
        L, a, b = lab[0] / 2.55, lab[1] - 128, lab[2] - 128

        # vector scaling to produce X, Y, Z
        y = (L + 16.) / 116.
        x = (a / 500.) + y
        z = y - (b / 200.)

        # merge back to get reconstructed XYZ color image
        out = torch.stack([x, y, z], dim=0)

        # apply boolean transforms
        mask = out > 0.2068966
        not_mask = torch.logical_not(mask)
        out.masked_scatter_(mask, torch.pow(torch.masked_select(out, mask), 3))
        out.masked_scatter_(not_mask, (torch.masked_select(out, not_mask) - 16 / 116) / 7.787)

        # rescale to the reference white (illuminant)
        out = torch.mul(out, self._white.type(out.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1))

        # convert XYZ -> RGB color domain
        arr = torch.tensordot(out, torch.t(self._xyz2rgb).type(out.dtype), dims=([0], [0]))
        mask = arr > 0.0031308
        not_mask = torch.logical_not(mask)
        arr.masked_scatter_(mask, 1.055 * torch.pow(torch.masked_select(arr, mask), 1 / 2.4) - 0.055)
        arr.masked_scatter_(not_mask, torch.masked_select(arr, not_mask) * 12.92)
        arr = arr.permute(2, 0, 1)
        return torch.clamp(arr, 0, 1)


@torch.no_grad()
def rgb2grey(img):
    img = img.detach().clone()
    img = 0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]
    img = img.unsqueeze(1)
    return img


@torch.no_grad()
class VBM_Converter:
    def __init__(self, device):
        self.device = device
        pass

    @torch.no_grad()
    def __call__(self, images_source_batch, images_target_batch, brighter=True, mode='hsv-s-w4'):
        B, _, _, _ = images_source_batch.shape
        images_source_converted_list = []
        for i in range(B):
            images_source = images_source_batch[i]
            torch.clamp(images_source, 0, 1)

            if brighter:
                images_source_converted = 1 - self.blur_filter(1 - images_source, mode='hsv-s-w4')
            else:
                images_source_converted = self.blur_filter(images_source, mode='hsv-s-w4')

            torch.clamp(images_source_converted, 0, 1)
            images_source_converted_list.append(images_source_converted)
        images_source2target = torch.stack(images_source_converted_list, dim=0)
        return images_source2target

    def DarkChannel(self, im):
        dc, _ = torch.min(im, dim=-1)
        return dc

    def AtmLight(self, im, dark):
        h, w = im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort(0)
        indices = indices[(imsz - numpx): imsz]

        atmsum = torch.zeros([1, 3]).cuda()
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def DarkIcA(self, im, A):
        im3 = torch.empty_like(im)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        return self.DarkChannel(im3)

    def get_saturation(self, im):
        saturation = (im.max(-1)[0] - im.min(-1)[0]) / (im.max(-1)[0] + 1e-10)
        return saturation

    def process(self, im, defog_A, IcA, mode='hsv-s-w4'):
        if mode == 'hsv-s-w4':
            img_s = self.get_saturation(im)
            s = (-img_s.mean() * 4).exp()
            param = torch.ones_like(img_s) * s
        else:
            raise NotImplementedError(f'{mode} not supported yet!')

        param = param[None, :, :, None]
        tx = 1 - param * IcA

        tx_1 = torch.tile(tx, [1, 1, 1, 3])
        return (im - defog_A[:, None, None, :]) / torch.maximum(tx_1, torch.tensor(0.01)) + defog_A[:, None, None, :]

    def blur_filter(self, X, mode):
        X = X.permute(1, 2, 0).contiguous()

        dark = self.DarkChannel(X)
        defog_A = self.AtmLight(X, dark)
        IcA = self.DarkIcA(X, defog_A)

        IcA = IcA.unsqueeze(-1)

        return self.process(X, defog_A, IcA, mode=mode)[0].permute(2, 0, 1).contiguous()
