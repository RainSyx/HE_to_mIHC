# convert_png_to_ometiff.py
import pyvips

def png_to_ometiff(
    png_path,
    out_path,
    tile=512,
    is_rgb=False,
    bit_depth=16,
):
    im = pyvips.Image.new_from_file(png_path, access="sequential")

    # 1. 去 alpha
    if im.hasalpha():
        im = im.flatten(background=0)

    # 2. 通道处理
    if is_rgb:
        # HE：必须是 3 通道
        if im.bands != 3:
            raise ValueError(f"RGB image expected 3 bands, got {im.bands}")
        im = im.cast("uchar")   # HE = 8-bit RGB
    else:
        # IF / DAPI：单通道
        if im.bands > 1:
            im = im[0]
        im = im.cast("ushort")  # IF = 16-bit

    # 3. 保存 pyramidal OME-TIFF
    im.tiffsave(
        out_path,
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=False,
        bigtiff=True,
        compression="lzw",
    )


if __name__ == "__main__":
    # HE
    # png_to_ometiff(
    #     "/home/guoxs/Syx/ROISE/ROSIE_datatry/HE_png/B2155897.png",
    #     "/home/guoxs/Syx/ROISE/VALIS/B2155897/slides/B2155897.ome.tif",
    #     is_rgb=True,
    # )

    # DAPI
    png_to_ometiff(
        "/home/guoxs/Syx/ROISE/ROSIE_datatry/codex_images/B2155897/HLA-DR.png",
        "/home/guoxs/Syx/ROISE/VALIS/B2155897/slides/B2155897_HLA-DR.ome.tif",
        is_rgb=False,
    )
