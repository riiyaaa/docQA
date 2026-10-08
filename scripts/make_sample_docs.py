"""Generate the fictional sample documents in sample_docs/.

Quillmere Appliances is an invented company. Every name, number, phone number
and policy below is made up so the evaluation set in eval/ has known answers.

The generated files are committed, so you only need this script to change them:
    pip install reportlab          # only needed to regenerate
    python scripts/make_sample_docs.py
    python eval/check_dataset.py   # confirm the questions still match
"""
from pathlib import Path

import docx
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "sample_docs"
NOTICE = "Quillmere Appliances is a fictional company. Test document for DocQA."
styles = getSampleStyleSheet()

# Each PDF is a list of pages; each page is a list of blocks.
# A block is ("h1"|"h2"|"p", text) or ("ul"|"ol", [items]).
WARRANTY = [
    [("h1", "Quillmere Limited Warranty Guide"),
     ("p", "Effective January 1, 2026"),
     ("h2", "1. Warranty Overview"),
     ("p", "This guide explains the Quillmere Limited Warranty for products purchased new from "
           "Quillmere or an authorized retailer in the United States or Canada."),
     ("ul", ["Major appliances (refrigerators, dishwashers, washers, dryers): 2 years parts and labor.",
             "Small appliances (air purifiers, blenders, coffee makers): 1 year parts and labor.",
             "Refrigerator compressor and sealed refrigeration system: 10 years, parts only. Labor for "
             "compressor repairs is covered only during the first 2 years."]),
     ("p", "Coverage begins on the original date of purchase. Keep your receipt or order confirmation; "
           "proof of purchase is required for every claim."),
     ("p", "Registration bonus: register your product at quillmere.example/register within 60 days of "
           "purchase and the standard parts-and-labor period is extended by 6 months. The bonus does not "
           "apply to compressor coverage or to refurbished products.")],
    [("h2", "2. What Is Covered"),
     ("p", "The warranty covers defects in materials and workmanship under normal household use."),
     ("p", "In-home service: for major appliances, a technician visits your home at no charge if you live "
           "within 75 miles of an authorized service center. Beyond 75 miles, a travel fee of $1.50 per "
           "mile applies to the distance over 75 miles."),
     ("p", "Small appliances are serviced by mail. Quillmere provides a prepaid shipping label, and repairs "
           "are completed within 10 business days of receipt."),
     ("p", "Replacement parts may be new or factory-refurbished. A part replaced under warranty is covered "
           "for 90 days or for the remainder of the original warranty, whichever is longer.")],
    [("h2", "3. What Is Not Covered"),
     ("ul", ["Cosmetic damage such as scratches, dents, or discoloration that does not affect operation.",
             "Consumable items, including water filters, air purifier filter cartridges, light bulbs, and batteries.",
             "Damage caused by improper installation, power surges, floods, or fire.",
             "Products used for commercial or rental purposes.",
             "Products with a removed or altered serial number.",
             "Products used outside the United States or Canada."]),
     ("p", "Parts sold by third-party sellers: the water inlet valve kit ZX-4410, sold by third-party "
           "marketplace sellers, is not approved by Quillmere. The ZX-4410 kit itself, and any damage it "
           "causes, are excluded from coverage. The genuine Quillmere water inlet valve, part number ZX-4401, "
           "is fully covered when installed by an authorized technician.")],
    [("h2", "4. How to File a Claim"),
     ("p", "File a claim within 30 days of discovering a defect."),
     ("ol", ["Find your model and serial number. On refrigerators, the label is inside the main door frame "
             "on the left side. On dishwashers, it is on the edge of the door. On air purifiers, it is on "
             "the back panel below the power cord.",
             "Submit a claim at claims.quillmere.example or call 1-800-555-0142.",
             "Upload proof of purchase and at least two photos of the problem.",
             "For major appliances, a technician is scheduled within 5 business days."]),
     ("p", "Quillmere sends a claim decision by email within 10 business days of receiving a complete claim.")],
    [("h2", "5. Repairs, Replacements, and Refunds"),
     ("p", "If the same defect cannot be fixed after 3 repair attempts, Quillmere will replace the product "
           "with the same or a comparable model, or refund the purchase price."),
     ("p", "Refunds are issued to the original payment method within 14 business days of approval. "
           "Customers in Canada are refunded in Canadian dollars."),
     ("p", "A replacement product is covered for the remainder of the original product's warranty, or "
           "1 year, whichever is longer."),
     ("p", "Escalations: if you disagree with a claim decision, email warranty-escalations@quillmere.example "
           "within 60 days of the decision. A senior specialist responds within 7 business days.")],
]

MANUAL = [
    [("h1", "Quillmere AirPure AP300 Air Purifier User Manual"),
     ("h2", "1. Specifications"),
     ("ul", ["Recommended room size: up to 450 sq ft (42 square meters).",
             "Clean air delivery rate (CADR): 280 cubic feet per minute for smoke.",
             "Noise level: 24 dB in Sleep mode to 52 dB in Turbo mode.",
             "Power consumption: 45 W.",
             "Replacement filter: AP-F300 three-stage filter (pre-filter, true HEPA, activated carbon).",
             "Fan modes: Sleep, Auto, and Turbo, plus 3 manual speeds.",
             "Weight: 7.2 kg."])],
    [("h2", "2. Setup and Operation"),
     ("p", "Before first use, open the back cover and remove the plastic wrap from the AP-F300 filter. "
           "The purifier will not start while the wrap is in place."),
     ("p", "Auto mode: the built-in PM2.5 sensor adjusts the fan speed to the air quality. The air quality "
           "ring glows blue (good), amber (moderate), or red (poor)."),
     ("p", "Sleep mode: the fan runs at its lowest speed and all display lights turn off."),
     ("p", "Child lock: press and hold the Mode button for 3 seconds to lock or unlock the controls. "
           "A lock icon appears on the display."),
     ("p", "Wi-Fi: connect the purifier to the Quillmere Home app to set schedules and check filter life.")],
    [("h2", "3. Cleaning and Maintenance"),
     ("p", "Pre-filter: vacuum the outer pre-filter every 2 weeks."),
     ("p", "Main filter: replace the AP-F300 filter every 6 to 8 months, or sooner in homes with pets or "
           "smokers. Do not wash the HEPA layer; water damages it."),
     ("p", "Reset the filter indicator: after installing a new filter, press and hold the Filter button "
           "for 5 seconds until the indicator turns off."),
     ("p", "Air quality sensor: clean the sensor lens on the side panel every 2 months with a dry cotton swab.")],
    [("h2", "4. Troubleshooting and Error Codes"),
     ("ul", ["E04: Fan motor fault. Unplug the purifier for 10 minutes, then restart it. If E04 returns, "
             "contact support.",
             "E11: Air quality sensor blocked or dirty. Clean the sensor lens with a dry cotton swab.",
             "E17: Filter life ended. Replace the AP-F300 filter and reset the filter indicator.",
             "E21: Filter not installed correctly. Reseat the filter and close the back cover firmly.",
             "Blinking red ring: the filter will reach end of life within 2 weeks. Order a replacement filter.",
             "Steady red ring: air quality is poor. This is normal and not an error."])],
]

# Distractor documents: similar products and policies whose details differ on
# purpose (same error codes with different meanings, similar part numbers,
# different warranty terms), so retrieval has to pick the right document.
AP200 = [
    [("h1", "Quillmere AirPure AP200 Air Purifier User Manual"),
     ("h2", "1. Specifications"),
     ("ul", ["Recommended room size: up to 300 sq ft.",
             "Clean air delivery rate (CADR): 190 cubic feet per minute for smoke.",
             "Noise level: 28 dB in Night mode to 55 dB in Max mode.",
             "Replacement filter: AP-F200 two-stage filter (true HEPA and activated carbon).",
             "Fan modes: Night, Auto, and Max."])],
    [("h2", "2. Operation and Maintenance"),
     ("p", "Child lock: press and hold the Power and Mode buttons together for 3 seconds."),
     ("p", "Night mode: the fan runs quietly and the display dims to 10 percent brightness."),
     ("p", "Replace the AP-F200 filter every 4 to 6 months. The filter light turns orange when a "
           "replacement is due. To reset it, hold the Filter button for 3 seconds.")],
    [("h2", "3. Error Codes"),
     ("ul", ["E04: Power supply fault. Use a different wall outlet. If E04 continues, contact support.",
             "E11: Air quality sensor fault. Contact support for a sensor replacement.",
             "E17: Fan speed sensor fault. Contact support. Do not replace the filter; the filter is not the cause.",
             "E21: Back cover open. Close the back cover until it clicks."])],
]

RF500 = [
    [("h1", "Quillmere RF500 French Door Refrigerator User Manual"),
     ("h2", "1. Specifications"),
     ("ul", ["Total capacity: 28 cubic feet.",
             "Ice production: up to 4 lb per day.",
             "Water filter: RF-W50. Replace the water filter every 6 months.",
             "Estimated energy use: 650 kWh per year."])],
    [("h2", "2. Temperature and Controls"),
     ("p", "Recommended settings: set the refrigerator to 37 degrees F (3 degrees C) and the freezer to "
           "0 degrees F (minus 18 degrees C)."),
     ("p", "Door alarm: an alarm sounds if a door stays open for more than 2 minutes."),
     ("p", "Control lock: press and hold the Lock button for 3 seconds to lock the control panel.")],
    [("h2", "3. Ice Maker and Water Dispenser"),
     ("p", "After installation, allow 24 hours for the first ice. Discard the first 3 batches of ice."),
     ("p", "If no ice is produced, check that the ice maker switch under the ice bin is set to ON and that "
           "the household water supply valve is fully open."),
     ("p", "After replacing the RF-W50 filter, dispense 3 gallons of water to flush air from the line.")],
    [("h2", "4. Error Codes"),
     ("ul", ["E04: Evaporator fan fault. Keep the doors closed and schedule service.",
             "E11: Freezer temperature sensor fault. Schedule service.",
             "E17: Ice maker fill tube frozen. Turn the ice maker off for 2 hours to let it thaw.",
             "E21: Water filter bypass detected. Install an RF-W50 filter."])],
]

DW700 = [
    [("h1", "Quillmere DW700 Dishwasher User Manual"),
     ("h2", "1. Specifications and Cycles"),
     ("ul", ["Capacity: 14 place settings.",
             "Noise level: 44 dBA.",
             "Normal cycle: 2 hours 10 minutes. Quick cycle: 60 minutes.",
             "Other cycles: Heavy and Eco. The upper rack height is adjustable."])],
    [("h2", "2. Detergent, Rinse Aid, and Cleaning"),
     ("p", "White spots or a cloudy film on dishes are usually caused by hard water. Raise the rinse aid "
           "setting to 4 or 5 (the scale runs from 1 to 5) and refill the rinse aid dispenser."),
     ("p", "Use detergent pods or powder made for automatic dishwashers. Clean the filter at the bottom of "
           "the tub once a month.")],
    [("h2", "3. Error Codes"),
     ("ul", ["E04: Heater fault. Schedule service.",
             "E11: Incoming water too hot. The inlet water temperature should be below 150 degrees F.",
             "E17: Water supply problem. Make sure the water supply valve is open.",
             "E21: Drain blocked. Clean the drain filter at the bottom of the tub and check the drain hose for kinks."])],
]

PRO = [
    [("h1", "Quillmere Pro Commercial Warranty"),
     ("h2", "1. Eligible Products and Coverage"),
     ("p", "This warranty applies only to Quillmere Pro commercial-series models, whose model names end in "
           "-PRO (for example, RF500-PRO). Consumer models used in businesses are not covered by this "
           "warranty or by the consumer warranty."),
     ("ul", ["All Pro models, including Pro refrigerators and Pro dishwashers: 1 year parts and labor.",
             "Pro refrigerator compressors: 5 years, parts only."]),
     ("p", "Pro products are not eligible for the consumer registration bonus.")],
    [("h2", "2. Service and Claims"),
     ("p", "On-site service is provided within 50 miles of an authorized commercial service center. "
           "Beyond 50 miles, a travel fee of $2.00 per mile applies."),
     ("p", "Submit claims through the Quillmere Pro portal using your business account. Quillmere responds "
           "within 2 business days, and a technician is dispatched within 48 hours of approval.")],
]

FAQ = [
    ("What are your customer support hours?",
     "Monday to Friday, 8 a.m. to 8 p.m. Central Time, and Saturday, 9 a.m. to 5 p.m. Central Time. "
     "We are closed on Sundays and US federal holidays."),
    ("Do you ship to Alaska and Hawaii?",
     "Yes. Standard ground shipping takes 7 to 10 business days. Small appliances ship at no extra charge; "
     "major appliances have a $149 delivery surcharge."),
    ("Can I extend my warranty?",
     "Yes. Quillmere Care+ adds 3 years of coverage after the standard warranty ends. You must buy it within "
     "1 year of the product's purchase date. Prices start at $89 for small appliances and $249 for major appliances."),
    ("Are refurbished products covered?",
     "Yes. Certified refurbished products include a 1-year parts-and-labor warranty, regardless of product "
     "category. The registration bonus does not apply."),
    ("Do you price match?",
     "We match the price of an identical new model sold by a major US retailer within 14 days of your purchase. "
     "Marketplace and auction sellers are excluded."),
    ("Where can I buy genuine replacement parts?",
     "Only from quillmere.example/parts or an authorized service center."),
]

POLICIES = f"""# Quillmere Online Store Policies

_{NOTICE}_

## Returns

- Unopened items: return within 30 days of delivery for a full refund.
- Opened items: return within 30 days of delivery; a 15% restocking fee applies.
- Major appliances that have been installed cannot be returned. Problems with installed appliances are handled under the Quillmere Limited Warranty.
- Gifts: returns are refunded as store credit.

## Shipping

- Free standard shipping on orders over $49.
- Express shipping (2 business days): $19.99.
- Major appliance delivery includes haul-away of your old appliance for $25.

## Price Adjustments

If the price of an item drops on quillmere.example within 14 days of your purchase, contact support to receive the difference.
"""

RELEASE_NOTES = f"""Quillmere AirPure AP300 - Firmware Release Notes
{NOTICE}

Version 2.1.0 (March 2, 2026)
- New: Schedule Turbo mode from the Quillmere Home app.
- Fixed: E11 sensor error appearing incorrectly in high humidity (above 80%).
- Known issue: Child lock turns off after a power outage. A fix is planned for version 2.1.1.

Version 2.0.3 (November 18, 2025)
- Improved: Sleep mode noise reduced from 26 dB to 24 dB.
- Improved: Faster Wi-Fi reconnection after router restarts.

Version 1.9.0 (June 10, 2025)
- New: Support for the Quillmere Home app.
"""


def write_pdf(path: Path, pages) -> None:
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(inch, 0.6 * inch, NOTICE)
        canvas.drawRightString(letter[0] - inch, 0.6 * inch, f"Page {doc.page}")
        canvas.restoreState()

    story = []
    for i, page in enumerate(pages):
        for kind, content in page:
            if kind in ("ul", "ol"):
                items = [ListItem(Paragraph(t, styles["BodyText"])) for t in content]
                story.append(ListFlowable(items, bulletType="bullet" if kind == "ul" else "1"))
            else:
                style = {"h1": "Title", "h2": "Heading2", "p": "BodyText"}[kind]
                story.append(Paragraph(content, styles[style]))
            story.append(Spacer(1, 6))
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=letter, topMargin=inch, bottomMargin=inch).build(
        story, onFirstPage=footer, onLaterPages=footer)


def write_faq(path: Path) -> None:
    d = docx.Document()
    d.add_heading("Quillmere Customer Support FAQ", level=1)
    d.add_paragraph(NOTICE)
    for q, a in FAQ:
        d.add_paragraph(q, style="Heading 3")
        d.add_paragraph(a)
    d.save(path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    write_pdf(OUT / "warranty_guide.pdf", WARRANTY)
    write_pdf(OUT / "ap300_manual.pdf", MANUAL)
    write_pdf(OUT / "ap200_manual.pdf", AP200)
    write_pdf(OUT / "rf500_manual.pdf", RF500)
    write_pdf(OUT / "dw700_manual.pdf", DW700)
    write_pdf(OUT / "pro_commercial_warranty.pdf", PRO)
    write_faq(OUT / "support_faq.docx")
    (OUT / "store_policies.md").write_text(POLICIES, encoding="utf-8")
    (OUT / "ap300_release_notes.txt").write_text(RELEASE_NOTES, encoding="utf-8")
    print(f"Wrote 9 files to {OUT}")
