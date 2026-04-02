"""
Seed Script: Pre-loads the FinQuery MITC content into the RAG system.
Parses the MITC text, chunks it, embeds it, and stores in ChromaDB + PostgreSQL.

Usage:
    python -m scripts.seed_mitc
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings
from app.db.session import init_database, _get_async_session_local
from app.db.mongodb import MongoDB
from app.models.schemas import IndexingConfig, ChunkStrategy
from app.rag.pipeline import rag_pipeline

settings = get_settings()


# ═══════════════════════════════════════════════════════════════
#  FINQUERY MITC RAW CONTENT
# ═══════════════════════════════════════════════════════════════

FINQUERY_MITC_CONTENT = """
MOST IMPORTANT TERMS & CONDITIONS - Version 1.76
FinQuery Partner Bank Credit Card

The Most Important Terms and Conditions are to be read along with the Card Member Agreement of FinQuery Partner Bank Ltd for complete understanding. The MITC are subject to change. Refer www.finquery.com for details.

SCHEDULE OF CHARGES

Interest Free Period: Up to 50 days.
Minimum Repayment Amount: 5% or minimum Rs 200.
Cash Advance Limit: 40% of the Credit Limit.

ANNUAL / RENEWAL FEES:
- Infinia: Rs 10,000
- Diners Black: Rs 10,000
- InterMiles HDFC Bank Diners Club: Rs 5,000
- Regalia / Business Regalia / Doctor's Regalia: Rs 2,500
- Diners Club Premium / Diners Privilege: Rs 2,500
- InterMiles HDFC Bank World / Signature: Rs 2,500
- 6E Rewards XL-IndiGo HDFC Bank: Rs 2,500
- Best Price Save Max: Rs 1,000
- Millennia / Regalia First / Business Regalia First: Rs 1,000
- Diners ClubMiles / AllMiles / Doctors Superia / Superia: Rs 1,000
- Diners Club Rewardz: Rs 1,000
- InterMiles HDFC Bank Platinum / Times Platinum Card: Rs 1,000
- Solitaire / Platinum Edge / MoneyBack / Business MoneyBack / MoneyBack Plus: Rs 500
- InterMiles Titanium & Select / Times Titanium Card / Easy EMI / Teacher's Platinum: Rs 500
- Business Platinum: Rs 299
- Small Business Moneyback: Rs 250
- Bharat CashBack / Freedom Card / IndianOil HDFC Bank Card: Rs 500
- Business Bharat CashBack / Business Freedom Card / Business Program: Rs 500
- 6E Rewards-IndiGo HDFC Bank: Rs 700

SPEND CONDITIONS FOR FEE WAIVER (excludes Cash on Call, Balance Transfer, Cash Withdrawal):
- Infinia / InterMiles Diners: Spend Rs 8 Lakh in a year for next year fee waiver
- Diners Black: Spend Rs 5 Lakh in a year
- Regalia / Business Regalia / Doctor's Regalia / Diners Club Premium / Diners Privilege: Spend Rs 3,00,000
- InterMiles HDFC Bank World / Signature: Spend Rs 3,00,000
- Millennia / AllMiles / Doctor's Superia / Diners Club Rewards / Regalia First / Diners ClubMiles: Spend Rs 1,00,000
- Superia Card: Spend Rs 75,000 in first year
- Best Price Save Max: Spend Rs 75,000
- Solitaire / Titanium Edge / Platinum Edge / MoneyBack / Business MoneyBack / MoneyBack Plus / Easy EMI / Best Price Save Smart: Spend Rs 50,000
- InterMiles HDFC Bank Titanium & Select / Times Titanium Card: Spend Rs 1,50,000
- InterMiles HDFC Bank Platinum / Times Platinum Card: Spend Rs 2,50,000
- Bharat / Business Freedom / Business Bharat CashBack / Bharat CashBack / IndianOil HDFC Bank Card: Spend Rs 50,000
- Small Business MoneyBack: Spend Rs 25,000
- Freedom: Spend Rs 50,000 annually
- Teacher's Platinum / Business Platinum: Spend Rs 50,000

Additional Card Fee: Lifetime Free

CHARGES ON REVOLVING CREDIT:
- Regalia / Business Regalia / Regalia First / Business Regalia First / Doctors Regalia / Millennia / MoneyBack / MoneyBack Plus / Indian Oil Card / Freedom / Times Titanium / Times Platinum / Others: 3.6% per month (43.2% annually) w.e.f. 1st Sep 2020
- Best Price Save Smart / Best Price Save Max: 3.49% per month (41.88% annually)
- InterMiles HDFC Bank Diners Club / Diners Black / Infinia: 1.99% per month (23.88% annually)
- Card issued against Fixed Deposit: 1.99% per month (23.88% annually)
- Business Program: maximum 3.6% per month (43.2% annually)

AUTO EMI CONVERSION (Easy EMI Credit Card):
Applicable on all transactions (excluding Fuel, Gold and Jewelry) of Rs 10,000 or more. EMI tenure: 9 months at ROI of 20%, processing fee Rs 99.

CASH ADVANCE CHARGES:
2.5% of amount withdrawn or Rs 500 whichever is higher. All cash advances attract revolving credit charges from date of withdrawal.

GRACE DAYS:
3 grace days from payment due date. Bank reports as past due to CICs only after 3 days.

LATE PAYMENT CHARGES (w.e.f. 1st Sep 2020):
- Less than Rs 100: Nil
- Rs 100 to Rs 500: Rs 100
- Rs 501 to Rs 5,000: Rs 500
- Rs 5,001 to Rs 10,000: Rs 600
- Rs 10,001 to Rs 25,000: Rs 800
- Rs 25,001 to Rs 50,000: Rs 1,100
- More than Rs 50,000: Rs 1,300

PAYMENT RETURN CHARGES: 2% of Payment amount subject to minimum of Rs 450.
OVERLIMIT CHARGES: 2.5% of overlimit amount, minimum Rs 550.
CASH PROCESSING FEE: Rs 100 for payments at HDFC Bank branches or ATMs.

FUEL TRANSACTION SURCHARGE:
Waived for transactions between Rs 400 to Rs 5,000.
- InterMiles HDFC Bank Diners Club / Infinia / Diners Club Black: Capped at Rs 1,000 every billing cycle (GST applicable)
- InterMiles World / Signature / Regalia / Diners Rewardz / Regalia First / Business Regalia First / InterMiles Platinum / AllMiles / Business Regalia / Doctor's Regalia / Times Platinum / 6E Rewards XL IndiGo Card: Capped at Rs 500 every billing cycle
- Millennia / MoneyBack / Freedom / MoneyBack Plus and all other cards: Capped at Rs 250 per billing cycle
- Fuel Surcharge Waiver is 1% of eligible fuel transaction amounts
- GST on fuel surcharge is non-refundable
- Reward Points / InterMiles will not be accrued on fuel transactions

LOAN PROCESSING FEES:
- InstaLoan: Rs 999 (exclusive of GST)
- Jumbo Loan: Rs 999 (exclusive of GST)
- Smart EMI: Rs 799 (exclusive of GST)
- Balance Transfer on EMI: 1% of Loan Amount, minimum Rs 250 (exclusive of GST)
- Loan PreClosure Charges: 3% of Balance Principal Outstanding

BALANCE TRANSFER PROCESSING: 1% of BT amount or Rs 250, whichever is higher.
REWARDS REDEMPTION FEE: Rs 99 per redemption request (not for Infinia/Diners Black, not for cashback).
DUPLICATE STATEMENT: Rs 100 per statement w.e.f. 1st Sep 2020.
REISSUE LOST/STOLEN/DAMAGED CARD: Rs 100.

FOREIGN CURRENCY TRANSACTIONS:
- Cross currency markup of 3.5% for most cards
- 2% for Regalia / Business Regalia / Doctor's Regalia / InterMiles World / Signature / InterMiles HDFC Bank Diners Club / Diners Privilege / Diners Club Premium / Regalia First / Business Regalia First / Infinia / Diners Black / Best Price Save Smart
- 2.5% for 6E Rewards XL-IndiGo HDFC Bank Card
- 3.0% for Diners Club Rewardz / Diners ClubMiles

GST: Applicable on all Fees, Interest and other Charges. CGST+SGST/UTGST for same state, IGST for different states.
TDS: 2% on aggregate cash withdrawals exceeding Rs 1 Crore per financial year.

RAILWAY TICKET PURCHASE FEE: 1% of transaction amount + GST (refer IRCTC website).

1. FEES AND CHARGES
The fees may vary for each Cardmember. Cash Advance fee is 2.5% (minimum Rs 500) of amount withdrawn. All cash advances attract revolving credit charges from date of withdrawal. HDFC Bank retains the right to alter any charges or fees from time to time with due intimation.

2. INTEREST FREE PERIOD
The interest free credit period ranges from 20 to 50 days subject to the scheme applicable. For example: HDFC Bank International Platinum Plus Card has billing date of 4th. Customer can spend from 5th April to 4th May, bill generated on 4th May, Payment Due Date is 24th May. Purchase on 14th April gets 41 days credit period. Purchase on 2nd May gets 23 days. Interest-free period only applies if all previous dues paid in full.

3. CREDIT LIMITS
HDFC Bank determines credit limit and cash withdrawal limit at sole discretion. Add-on Cardmembers share the same limit. Limits communicated at card delivery and in monthly statements. Bank may decrease credit limit based on internal criteria with immediate notification. Cardmembers can request increases by writing with income documents.

4. FINANCE CHARGES
Finance charges are calculated as: (outstanding amount x 3.6% per month x 12 months x number of days) / 365.

Example calculation:
Billing date 18th of month. Rs 15,000 purchase on 10 April, Rs 5,000 on 15 April. Total due Rs 20,000.
Payment Rs 2,000 on 12 May (late), Rs 15,000 on 15 May.
Interest on Rs 15,000 @ 3.6% pm from 19 April to 11 May (23 days) = Rs 408.33
Interest on Rs 13,000 @ 3.6% pm from 12 May to 14 May (3 days) = Rs 46.16
Interest on Rs 5,000 @ 3.6% pm from 19 April to 14 May (26 days) = Rs 153.86
Interest on Rs 3,000 @ 3.6% pm from 15 May to 18 May (4 days) = Rs 14.20
Interest on Rs 1,000 fresh spends @ 3.6% pm from 15 May to 18 May (5 days) = Rs 5.92
Total interest = Rs 628.47. Plus Late Payment Charges. Plus GST @ 18%.

MINIMUM AMOUNT DUE (MAD): 5% of Total Amount Due, rounded to next higher 10th digit. If Rs 200 or lower, full amount is minimum.

5. PAYMENT METHODS
- NetBanking / ATMs / PhoneBanking (HDFC account holders)
- Standing Instruction (auto-debit from HDFC Bank account on due date)
- NEFT / RTGS / IMPS from other banks (IFSC code HDFC0000128)
- BillDesk (other banks' NetBanking)
- Cheque/Draft at HDFC Bank branches (3 working days before due date, 5 for outstation)
- Cash at branches

Payment adjustment order: taxes, fees/charges, interest, cash advances, purchases.

6. BILLING DISPUTES
Report discrepancies within 30 days of statement date. Bank may reverse charge temporarily pending investigation. Documents provided within 30 days per Visa/MasterCard/Diners guidelines. GST not reversed on any disputed fees/charges/interest.

7. DEFAULT AND DEBT COLLECTION
Reminders via post, fax, telephone, e-mail, SMS, and/or third-party agents (who must follow code of conduct). Bank has right of lien on deposits and assets held with the bank.

8. CARD TERMINATION
Cardmember can terminate by writing to HDFC Bank. All add-on cards also terminated. No pro-rata fee refund. Bank may restrict/terminate for unusual patterns, excessive utilization, merchant collusion, or undue reward accumulation. Death/incapacitation automatically cancels all cards.

9. LOSS / THEFT / MISUSE
Notify 24-hour call center immediately. Don't use card if found after reporting. Add-on card loss invalidates primary and all other add-ons (and vice versa). No bank liability for transactions before reporting. Zero liability after proper notification. File police complaint and provide copy to bank. If Cardmember acted fraudulently, liable for all losses.

10. REWARD POINTS
Points earned per Rs 150 spent (multiples). Below Rs 150 or residual amounts don't earn points. Silver/Freedom Plus: per Rs 200. No points on EasyEMI, e-wallet loading, fuel transactions. Points reversed on SmartEMI conversion. Insurance points capped: 5,000/day for Infinia/Diners Black, 2,000/day for others.

Validity: Most cards - 2 years from accumulation. Infinia/Diners Black - 3 years. EasyEMI - 1 year. Card unused >365 days = points forfeited. Card blocked/hotlisted and not reactivated within 6 months = points nullified.

Redemption: Rs 99 fee per request (free for Infinia/Diners Black, and for cashback). Flight/hotel on Smartbuy: Infinia/Black up to 70% via points, others 50%. Cashback adjusts next statement balance, not counted as payment.

11. CONTACTLESS PAYMENTS
Tap-to-pay for transactions up to Rs 5,000 - no PIN/signature needed. Same security as chip/PIN.

12. LOUNGE ACCESS
Priority Pass: $27 + taxes per visit beyond complimentary cap. Infinia gets unlimited complimentary access. Guests always $27 + taxes. Diners Club validation: Rs 2 (non-refundable). Priority Pass withdrawn if card inactive >90 days.

13. DISCLOSURE / CIBIL
Bank reports to CIBIL and credit bureaus per Credit Information Companies Regulation Act, 2005. Data refresh within 60 days of payment on overdue accounts. Bank may record conversations.

14. WALLET LOADING
Wallet loading via credit card: 2.5% of load value. No reward points on wallet loading.

15. CARD RESTRICTIONS
Card must not be used for: Forex trading, lottery, betting, gambling, dating. Violations may attract FEMA action. Unused >1 year: bank will intimate for reactivation or close.

16. GRIEVANCE REDRESSAL
Write to: Manager, Grievance Cell, HDFC Bank Credit Cards Division, 8 Lattice Bridge Road, Thiruvanmiyur, Chennai 600041.
Phone: 044-23625600 (9:30 AM to 5:30 PM, Mon-Fri).
Toll-free: 1800 266 4332 for select cities.

MITC available in 13 regional languages on hdfcbank.com. English version prevails in case of inconsistency.
"""


async def seed_mitc():
    """Seed the HDFC MITC content into the RAG system."""
    print("=" * 60)
    print("  HDFC MITC — RAG Seed Script")
    print("=" * 60)

    # 1. Initialize databases (Postgres & MongoDB)
    print("\n[1/4] Initializing database...")
    await init_database()
    if settings.database_type == "mongodb":
        await MongoDB.connect()

    # 2. Initialize RAG pipeline
    print("[2/4] Initializing RAG pipeline (embedding model + ChromaDB)...")
    rag_pipeline.initialize()

    # 3. Index the MITC content
    print("[3/4] Indexing HDFC MITC document...")
    config = IndexingConfig(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=512,
        chunk_overlap=50,
    )

    AsyncSessionLocal = _get_async_session_local()
    async with AsyncSessionLocal() as db:
        result = await rag_pipeline.index_raw_text(
            text=FINQUERY_MITC_CONTENT,
            title="FinQuery Partner Bank Credit Card MITC v1.76",
            db=db,
            config=config,
        )

    # 4. Report results
    print(f"\n[4/4] Indexing Complete!")
    print(f"  • Document ID:      {result.document_id}")
    print(f"  • Status:           {result.status}")
    print(f"  • Chunks created:   {result.chunks_created}")
    print(f"  • Embeddings stored: {result.embeddings_stored}")
    print(f"  • Processing time:  {result.processing_time_ms:.0f}ms")

    if result.errors:
        print(f"  ⚠️  Errors: {result.errors}")
    else:
        print(f"\n  ✅ FinQuery MITC successfully indexed! Start the server:")
        print(f"     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print(f"     Then open http://localhost:8000")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(seed_mitc())
