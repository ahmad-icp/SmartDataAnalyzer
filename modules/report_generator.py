from reporting import generate_html_report, html_to_pdf_bytes


def generate_report_html(profile_html: str, extra_html: str = "") -> str:
    return generate_html_report(profile_html, extra_html)


__all__ = ["generate_report_html", "html_to_pdf_bytes"]
