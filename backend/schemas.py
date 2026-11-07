"""The HTTP input contract: every model input must be supplied explicitly."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Binary = Literal[0, 1]


class Transaction(BaseModel):
    """Validate the same 14 inputs used by the trained notebook model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    timestamp: AwareDatetime = Field(description="ISO 8601 time with Z or an explicit UTC offset")
    amount: NonNegative = Field(description="Transaction amount in USD, the training currency")
    merchant_category: str = Field(min_length=1, max_length=80)
    merchant_country: str = Field(pattern=r"^[A-Za-z]{2}$")
    card_present: Binary
    device_type: str = Field(min_length=1, max_length=80)
    device_known: Binary
    ip_risk_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    is_foreign_txn: Binary
    time_since_last_s: NonNegative
    velocity_1h: int = Field(ge=0)
    amount_vs_avg_ratio: NonNegative
    account_age_days: NonNegative
    has_2fa: Binary

    @field_validator("merchant_country")
    @classmethod
    def normalize_country(cls, value: str) -> str:
        """Use the uppercase country codes seen during training."""
        return value.upper()

    def to_record(self) -> dict:
        """Convert the validated date to JSON text for the notebook's predictor."""
        return self.model_dump(mode="json")
