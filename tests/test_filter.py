import re

def test_page_number_filtering():
    """Verify image filename page extraction and filtering."""
    referenced_images = [
        "chart_Apple 10-K PDF 2023_p24_1_8071c5.png",
        "table_Apple 10-K PDF 2023_p24_1_03efef.png",
        "chart_Apple 10-K PDF 2023_p30_1_d05266.png",
        "table_Apple 10-K PDF 2023_p30_1_bf3983.png"
    ]
    relevant_pages = {24}

    filtered_images = []
    for img in referenced_images:
        match = re.search(r'_p(\d+)_', img)
        if match:
            img_page = int(match.group(1))
            if img_page in relevant_pages:
                filtered_images.append(img)
        else:
            filtered_images.append(img)

    assert len(filtered_images) == 2
    assert "chart_Apple 10-K PDF 2023_p24_1_8071c5.png" in filtered_images
    assert "table_Apple 10-K PDF 2023_p24_1_03efef.png" in filtered_images

