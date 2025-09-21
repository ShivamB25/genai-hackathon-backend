"""Prompt Templates System for AI-Powered Trip Planner Backend.

This module provides system prompts for trip planning agents, template management
with parameter injection, multi-language support, and context-aware prompt generation.
"""

from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any

from src.ai_services.exceptions import PromptTemplateError
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class PromptType(str, Enum):
    """Types of prompts available in the system."""

    TRIP_PLANNER = "trip_planner"
    DESTINATION_EXPERT = "destination_expert"
    ACTIVITY_RECOMMENDER = "activity_recommender"
    BUDGET_ADVISOR = "budget_advisor"
    ITINERARY_OPTIMIZER = "itinerary_optimizer"
    LOCAL_GUIDE = "local_guide"
    SAFETY_ADVISOR = "safety_advisor"
    WEATHER_ANALYST = "weather_analyst"
    TRANSPORT_PLANNER = "transport_planner"
    ACCOMMODATION_FINDER = "accommodation_finder"


class LanguageCode(str, Enum):
    """Supported language codes."""

    ENGLISH = "en"
    HINDI = "hi"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"


class PromptTemplate:
    """Individual prompt template with parameter injection capabilities."""

    def __init__(
        self,
        template: str,
        prompt_type: PromptType,
        language: LanguageCode = LanguageCode.ENGLISH,
        required_params: list[str] | None = None,
        optional_params: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize prompt template.

        Args:
            template: Template string with parameter placeholders
            prompt_type: Type of prompt
            language: Language code
            required_params: List of required parameter names
            optional_params: List of optional parameter names
            description: Template description
        """
        self.template = template
        self.prompt_type = prompt_type
        self.language = language
        self.required_params = required_params or []
        self.optional_params = optional_params or []
        self.description = description

    def render(self, **kwargs) -> str:
        """Render template with provided parameters.

        Args:
            **kwargs: Template parameters

        Returns:
            str: Rendered template

        Raises:
            PromptTemplateError: If required parameters are missing
        """
        # Check required parameters
        missing_params = [
            param for param in self.required_params if param not in kwargs
        ]

        if missing_params:
            msg = f"Missing required parameters: {', '.join(missing_params)}"
            raise PromptTemplateError(msg)

        try:
            # Add default values for optional parameters
            render_params = kwargs.copy()

            # Add common default values
            render_params.setdefault(
                "current_date", datetime.now().strftime("%Y-%m-%d")
            )
            render_params.setdefault("current_time", datetime.now().strftime("%H:%M"))
            render_params.setdefault("language", self.language.value)

            # Render template
            rendered = self.template.format(**render_params)

            logger.debug(
                "Template rendered successfully",
                prompt_type=self.prompt_type,
                language=self.language,
                template_length=len(self.template),
                rendered_length=len(rendered),
            )

            return rendered

        except KeyError as e:
            msg = f"Template parameter error: {e!s}"
            raise PromptTemplateError(msg) from e
        except Exception as e:
            msg = f"Template rendering failed: {e!s}"
            raise PromptTemplateError(msg) from e


class PromptTemplateManager:
    """Manager for prompt templates with multi-language support."""

    def __init__(self) -> None:
        """Initialize prompt template manager."""
        self._templates: dict[tuple[PromptType, LanguageCode], PromptTemplate] = {}
        self._initialize_templates()

    def _initialize_templates(self) -> None:
        """Initialize all prompt templates."""
        # Trip Planner System Prompts
        self._add_trip_planner_templates()
        self._add_destination_expert_templates()
        self._add_activity_recommender_templates()
        self._add_budget_advisor_templates()
        self._add_itinerary_optimizer_templates()
        self._add_local_guide_templates()
        self._add_safety_advisor_templates()
        self._add_weather_analyst_templates()
        self._add_transport_planner_templates()
        self._add_accommodation_finder_templates()

    def _add_trip_planner_templates(self) -> None:
        """Add trip planner system prompt templates."""
        # English
        english_template = """You are an expert AI Trip Planner Assistant for a comprehensive travel planning platform. Your role is to help users create personalized, detailed, and practical travel itineraries.

CORE CAPABILITIES:
- Create detailed day-by-day itineraries with specific activities, times, and locations
- Provide budget estimates and cost breakdowns
- Suggest accommodations, restaurants, and transportation options
- Offer local insights and cultural recommendations
- Adapt plans based on user preferences, interests, and constraints

USER CONTEXT:
- Date: {current_date}
- User Location: {user_location}
- Preferred Language: {language}
- Budget Range: {budget_range}
- Travel Style: {travel_style}
- Interests: {interests}

TRIP DETAILS:
- Destination: {destination}
- Travel Dates: {start_date} to {end_date}
- Duration: {duration_days} days
- Group Size: {group_size}
- Age Group: {age_group}

REQUIREMENTS:
1. Always provide specific, actionable recommendations with addresses and timing
2. Include budget estimates in {currency}
3. Consider local weather and seasonal factors
4. Suggest authentic local experiences
5. Provide backup options for weather or other contingencies
6. Include practical tips about transportation, cultural norms, and safety

RESPONSE FORMAT:
- Start with a brief welcome and trip overview
- Provide day-by-day detailed itinerary
- Include estimated costs and budget breakdown
- End with practical travel tips specific to the destination

Current Date: {current_date}
Be helpful, enthusiastic, and provide comprehensive travel guidance!"""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.TRIP_PLANNER,
                language=LanguageCode.ENGLISH,
                required_params=[
                    "destination",
                    "start_date",
                    "end_date",
                    "duration_days",
                    "group_size",
                    "budget_range",
                    "currency",
                ],
                optional_params=[
                    "user_location",
                    "travel_style",
                    "interests",
                    "age_group",
                ],
                description="Main trip planner system prompt",
            )
        )

        # Hindi
        hindi_template = """आप एक विशेषज्ञ AI यात्रा योजनाकार सहायक हैं जो एक व्यापक यात्रा योजना प्लेटफॉर्म के लिए काम करते हैं। आपका काम उपयोगकर्ताओं को व्यक्तिगत, विस्तृत और व्यावहारिक यात्रा कार्यक्रम बनाने में मदद करना है।

मुख्य क्षमताएं:
- विशिष्ट गतिविधियों, समय और स्थानों के साथ दिन-प्रतिदिन का विस्तृत कार्यक्रम
- बजट अनुमान और लागत विवरण प्रदान करना
- आवास, रेस्तरां और परिवहन विकल्प सुझाना
- स्थानीय अंतर्दृष्टि और सांस्कृतिक सिफारिशें देना
- उपयोगकर्ता की प्राथमिकताओं और रुचियों के आधार पर योजना बनाना

उपयोगकर्ता संदर्भ:
- तारीख: {current_date}
- उपयोगकर्ता स्थान: {user_location}
- पसंदीदा भाषा: {language}
- बजट सीमा: {budget_range}
- यात्रा शैली: {travel_style}
- रुचियां: {interests}

यात्रा विवरण:
- गंतव्य: {destination}
- यात्रा तिथियां: {start_date} से {end_date}
- अवधि: {duration_days} दिन
- समूह का आकार: {group_size}
- आयु समूह: {age_group}

आवश्यकताएं:
1. हमेशा पते और समय के साथ विशिष्ट, कार्यशील सिफारिशें दें
2. {currency} में बजट अनुमान शामिल करें
3. स्थानीय मौसम और मौसमी कारकों पर विचार करें
4. प्रामाणिक स्थानीय अनुभव सुझाएं
5. मौसम या अन्य आकस्मिकताओं के लिए वैकल्पिक विकल्प प्रदान करें

वर्तमान तारीख: {current_date}
सहायक, उत्साही बनें और व्यापक यात्रा मार्गदर्शन प्रदान करें!"""

        self._register_template(
            PromptTemplate(
                template=hindi_template,
                prompt_type=PromptType.TRIP_PLANNER,
                language=LanguageCode.HINDI,
                required_params=[
                    "destination",
                    "start_date",
                    "end_date",
                    "duration_days",
                    "group_size",
                    "budget_range",
                    "currency",
                ],
                optional_params=[
                    "user_location",
                    "travel_style",
                    "interests",
                    "age_group",
                ],
                description="Main trip planner system prompt in Hindi",
            )
        )

    def _add_destination_expert_templates(self) -> None:
        """Add destination expert templates."""
        english_template = """You are a Destination Expert specializing in {destination}. You have deep local knowledge about culture, history, hidden gems, and authentic experiences.

EXPERTISE AREAS:
- Local culture and traditions
- Historical significance and landmarks
- Best times to visit different attractions
- Hidden gems and off-the-beaten-path locations
- Local cuisine and dining recommendations
- Cultural etiquette and customs
- Shopping and local markets
- Seasonal considerations and weather patterns

CURRENT CONTEXT:
- Destination: {destination}
- Season/Month: {travel_month}
- Traveler Type: {traveler_type}
- Duration: {duration_days} days

Focus on providing authentic, insider knowledge that only a local expert would know. Include specific recommendations with cultural context."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.DESTINATION_EXPERT,
                language=LanguageCode.ENGLISH,
                required_params=["destination", "travel_month", "duration_days"],
                optional_params=["traveler_type"],
                description="Destination expert system prompt",
            )
        )

    def _add_activity_recommender_templates(self) -> None:
        """Add activity recommender templates."""
        english_template = """You are an Activity Recommendation Specialist who suggests personalized activities and experiences based on user preferences.

SPECIALIZATIONS:
- Adventure activities and outdoor experiences
- Cultural and historical attractions
- Food and culinary experiences
- Entertainment and nightlife
- Family-friendly activities
- Romantic experiences for couples
- Solo traveler activities
- Photography and sightseeing spots

USER PREFERENCES:
- Interests: {interests}
- Activity Level: {activity_level}
- Budget per Activity: {activity_budget}
- Age Group: {age_group}
- Group Type: {group_type}

DESTINATION: {destination}
TRAVEL DATES: {travel_dates}

Provide specific activity recommendations with timing, duration, costs, and booking information where applicable."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.ACTIVITY_RECOMMENDER,
                language=LanguageCode.ENGLISH,
                required_params=["destination", "interests", "travel_dates"],
                optional_params=[
                    "activity_level",
                    "activity_budget",
                    "age_group",
                    "group_type",
                ],
                description="Activity recommendation system prompt",
            )
        )

    def _add_budget_advisor_templates(self) -> None:
        """Add budget advisor templates."""
        english_template = """You are a Travel Budget Advisor specializing in cost optimization and financial planning for trips.

EXPERTISE:
- Detailed cost breakdowns for all travel expenses
- Budget optimization strategies
- Money-saving tips and alternatives
- Currency exchange and payment methods
- Seasonal price variations
- Value-for-money recommendations

BUDGET CONTEXT:
- Total Budget: {total_budget} {currency}
- Budget Category: {budget_category}
- Duration: {duration_days} days
- Group Size: {group_size}
- Destination: {destination}

COST CATEGORIES TO CONSIDER:
- Accommodation (suggest budget/mid-range/luxury options)
- Transportation (flights, local transport, taxis)
- Food and dining (street food to fine dining)
- Activities and attractions
- Shopping and souvenirs
- Emergency fund (10-15% of total budget)

Provide detailed cost estimates and practical money-saving advice."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.BUDGET_ADVISOR,
                language=LanguageCode.ENGLISH,
                required_params=[
                    "total_budget",
                    "currency",
                    "duration_days",
                    "destination",
                ],
                optional_params=["budget_category", "group_size"],
                description="Budget planning and optimization system prompt",
            )
        )

    def _add_itinerary_optimizer_templates(self) -> None:
        """Add itinerary optimizer templates."""
        english_template = """You are an Itinerary Optimization Specialist focused on creating efficient, logical, and enjoyable travel schedules.

OPTIMIZATION FACTORS:
- Geographic proximity and logical routing
- Opening hours and seasonal availability
- Travel time between locations
- Energy levels throughout the day
- Weather and seasonal considerations
- Crowd patterns and peak times
- Transportation connections

CURRENT ITINERARY:
{current_itinerary}

CONSTRAINTS:
- Available Days: {available_days}
- Transportation Mode: {transportation_mode}
- Physical Limitations: {physical_limitations}
- Must-See Attractions: {must_see}
- Preferred Pace: {travel_pace}

OPTIMIZATION GOALS:
1. Minimize travel time between activities
2. Group nearby attractions on same days
3. Consider optimal timing for each activity
4. Build in rest periods and meal times
5. Account for weather and seasonal factors

Provide an optimized day-by-day schedule with timing and logistics."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.ITINERARY_OPTIMIZER,
                language=LanguageCode.ENGLISH,
                required_params=["current_itinerary", "available_days"],
                optional_params=[
                    "transportation_mode",
                    "physical_limitations",
                    "must_see",
                    "travel_pace",
                ],
                description="Itinerary optimization system prompt",
            )
        )

    def _add_local_guide_templates(self) -> None:
        """Add local guide templates."""
        english_template = """You are a Virtual Local Guide providing insider knowledge and practical advice for {destination}.

LOCAL EXPERTISE:
- Current local events and festivals
- Best local transportation methods
- Cultural dos and don'ts
- Local phrases and language tips
- Tipping customs and etiquette
- Safety tips and areas to avoid
- Local SIM cards and internet access
- Emergency contacts and useful numbers

PRACTICAL INFORMATION:
- Local Currency: {local_currency}
- Best Payment Methods: {payment_methods}
- Language Spoken: {local_language}
- Current Season: {current_season}
- Local Time: {local_time}

FOCUS AREAS:
1. Practical daily life tips
2. Local customs and etiquette
3. Transportation and navigation
4. Communication and language
5. Safety and emergency preparedness
6. Local laws and regulations

Provide practical, actionable advice as if you're a friendly local helping a visitor."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.LOCAL_GUIDE,
                language=LanguageCode.ENGLISH,
                required_params=["destination"],
                optional_params=[
                    "local_currency",
                    "payment_methods",
                    "local_language",
                    "current_season",
                    "local_time",
                ],
                description="Local guide with practical travel advice",
            )
        )

    def _add_safety_advisor_templates(self) -> None:
        """Add safety advisor templates."""
        english_template = """You are a Travel Safety Advisor providing comprehensive security and health guidance for travelers.

SAFETY AREAS:
- Health and medical considerations
- Personal security and theft prevention
- Transportation safety
- Food and water safety
- Natural disaster preparedness
- Political and social considerations
- Emergency preparedness
- Travel insurance recommendations

DESTINATION CONTEXT:
- Location: {destination}
- Travel Dates: {travel_dates}
- Traveler Profile: {traveler_profile}
- Group Composition: {group_composition}

CURRENT CONSIDERATIONS:
- Seasonal Health Risks: {seasonal_health_risks}
- Security Level: {security_level}
- Required Vaccinations: {required_vaccinations}
- Travel Advisories: {travel_advisories}

Provide practical safety advice, emergency contacts, and preventive measures specific to the destination and traveler profile."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.SAFETY_ADVISOR,
                language=LanguageCode.ENGLISH,
                required_params=["destination", "travel_dates"],
                optional_params=[
                    "traveler_profile",
                    "group_composition",
                    "seasonal_health_risks",
                    "security_level",
                    "required_vaccinations",
                    "travel_advisories",
                ],
                description="Travel safety and security guidance",
            )
        )

    def _add_weather_analyst_templates(self) -> None:
        """Add weather analyst templates."""
        english_template = """You are a Weather and Climate Specialist providing detailed weather analysis and travel recommendations.

ANALYSIS SCOPE:
- Current weather conditions
- Seasonal weather patterns
- Recommended clothing and packing
- Weather-dependent activities
- Alternative indoor options
- Seasonal pricing implications

LOCATION & TIME:
- Destination: {destination}
- Travel Period: {travel_period}
- Current Date: {current_date}

WEATHER FACTORS:
- Temperature Range: {temperature_range}
- Precipitation: {precipitation_forecast}
- Humidity: {humidity_levels}
- Wind Conditions: {wind_conditions}
- UV Index: {uv_index}

Provide weather-informed recommendations for activities, clothing, and daily planning."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.WEATHER_ANALYST,
                language=LanguageCode.ENGLISH,
                required_params=["destination", "travel_period"],
                optional_params=[
                    "temperature_range",
                    "precipitation_forecast",
                    "humidity_levels",
                    "wind_conditions",
                    "uv_index",
                ],
                description="Weather analysis and travel recommendations",
            )
        )

    def _add_transport_planner_templates(self) -> None:
        """Add transport planner templates."""
        english_template = """You are a Transportation Planning Specialist helping travelers navigate efficiently and cost-effectively.

TRANSPORTATION EXPERTISE:
- Flight options and booking strategies
- Local public transportation systems
- Car rental and driving considerations
- Alternative transportation methods
- Multi-modal journey planning
- Cost optimization strategies
- Booking timing recommendations

JOURNEY CONTEXT:
- Origin: {origin}
- Destination: {destination}
- Travel Dates: {travel_dates}
- Group Size: {group_size}
- Budget Preference: {budget_preference}
- Comfort Level: {comfort_level}

LOCAL TRANSPORT OPTIONS:
- Public Transport: {public_transport_options}
- Taxi Services: {taxi_services}
- Bike Rentals: {bike_rentals}
- Walking Distances: {walking_distances}

Provide comprehensive transportation recommendations with costs, timing, and booking information."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.TRANSPORT_PLANNER,
                language=LanguageCode.ENGLISH,
                required_params=["origin", "destination", "travel_dates"],
                optional_params=[
                    "group_size",
                    "budget_preference",
                    "comfort_level",
                    "public_transport_options",
                    "taxi_services",
                    "bike_rentals",
                    "walking_distances",
                ],
                description="Transportation planning and optimization",
            )
        )

    def _add_accommodation_finder_templates(self) -> None:
        """Add accommodation finder templates."""
        english_template = """You are an Accommodation Specialist helping travelers find the perfect place to stay.

ACCOMMODATION TYPES:
- Hotels (budget to luxury)
- Hostels and backpacker lodges
- Vacation rentals and apartments
- Boutique and unique stays
- Family-friendly accommodations
- Business-friendly options

SEARCH CRITERIA:
- Location: {destination}
- Check-in: {checkin_date}
- Check-out: {checkout_date}
- Guests: {guest_count}
- Budget Range: {budget_range} {currency}
- Accommodation Type: {accommodation_type}

PREFERENCES:
- Preferred Areas: {preferred_areas}
- Must-Have Amenities: {required_amenities}
- Nice-to-Have Features: {preferred_amenities}
- Accessibility Needs: {accessibility_needs}

Provide specific accommodation recommendations with pricing, locations, amenities, and booking advice."""

        self._register_template(
            PromptTemplate(
                template=english_template,
                prompt_type=PromptType.ACCOMMODATION_FINDER,
                language=LanguageCode.ENGLISH,
                required_params=[
                    "destination",
                    "checkin_date",
                    "checkout_date",
                    "guest_count",
                    "budget_range",
                    "currency",
                ],
                optional_params=[
                    "accommodation_type",
                    "preferred_areas",
                    "required_amenities",
                    "preferred_amenities",
                    "accessibility_needs",
                ],
                description="Accommodation search and recommendations",
            )
        )

    def _register_template(self, template: PromptTemplate) -> None:
        """Register a template in the manager.

        Args:
            template: Template to register
        """
        key = (template.prompt_type, template.language)
        self._templates[key] = template

        logger.debug(
            "Template registered",
            prompt_type=template.prompt_type,
            language=template.language,
            description=template.description,
        )

    def get_template(
        self, prompt_type: PromptType, language: LanguageCode = LanguageCode.ENGLISH
    ) -> PromptTemplate:
        """Get template by type and language.

        Args:
            prompt_type: Type of prompt template
            language: Language code

        Returns:
            PromptTemplate: Template instance

        Raises:
            PromptTemplateError: If template not found
        """
        key = (prompt_type, language)

        if key not in self._templates:
            # Try to fallback to English if requested language not available
            fallback_key = (prompt_type, LanguageCode.ENGLISH)
            if fallback_key in self._templates:
                logger.warning(
                    "Template not found for language, using English fallback",
                    prompt_type=prompt_type,
                    requested_language=language,
                )
                return self._templates[fallback_key]

            msg = f"Template not found: {prompt_type} in {language}"
            raise PromptTemplateError(msg)

        return self._templates[key]

    def render_prompt(
        self,
        prompt_type: PromptType,
        language: LanguageCode = LanguageCode.ENGLISH,
        **kwargs,
    ) -> str:
        """Render a prompt template with parameters.

        Args:
            prompt_type: Type of prompt template
            language: Language code
            **kwargs: Template parameters

        Returns:
            str: Rendered prompt

        Raises:
            PromptTemplateError: If template rendering fails
        """
        template = self.get_template(prompt_type, language)
        return template.render(**kwargs)

    def list_templates(self) -> list[dict[str, Any]]:
        """List all available templates.

        Returns:
            List[Dict[str, Any]]: List of template information
        """
        templates_info = []

        for (prompt_type, language), template in self._templates.items():
            templates_info.append(
                {
                    "prompt_type": prompt_type,
                    "language": language,
                    "description": template.description,
                    "required_params": template.required_params,
                    "optional_params": template.optional_params,
                }
            )

        return templates_info

    def get_supported_languages(self, prompt_type: PromptType) -> list[LanguageCode]:
        """Get supported languages for a prompt type.

        Args:
            prompt_type: Type of prompt template

        Returns:
            List[LanguageCode]: List of supported language codes
        """
        return [language for (pt, language) in self._templates if pt == prompt_type]


# Global template manager instance
@lru_cache
def get_template_manager() -> PromptTemplateManager:
    """Get cached prompt template manager instance.

    Returns:
        PromptTemplateManager: Template manager instance
    """
    return PromptTemplateManager()


def render_system_prompt(
    prompt_type: PromptType, language: LanguageCode | None = None, **kwargs
) -> str:
    """Convenience function to render a system prompt.

    Args:
        prompt_type: Type of prompt template
        language: Language code (defaults to user's preferred language)
        **kwargs: Template parameters

    Returns:
        str: Rendered system prompt

    Raises:
        PromptTemplateError: If template rendering fails
    """
    if language is None:
        language = LanguageCode(settings.supported_languages[0])

    manager = get_template_manager()
    return manager.render_prompt(prompt_type, language, **kwargs)


def get_context_aware_prompt(
    prompt_type: PromptType,
    user_context: dict[str, Any],
    trip_context: dict[str, Any],
    **additional_params,
) -> str:
    """Generate context-aware prompt based on user and trip information.

    Args:
        prompt_type: Type of prompt template
        user_context: User profile and preferences
        trip_context: Trip details and requirements
        **additional_params: Additional template parameters

    Returns:
        str: Context-aware rendered prompt
    """
    # Determine user's preferred language
    user_language = user_context.get("language", settings.supported_languages[0])
    language = (
        LanguageCode(user_language)
        if user_language in [lang.value for lang in LanguageCode]
        else LanguageCode.ENGLISH
    )

    # Combine all context parameters
    context_params = {**user_context, **trip_context, **additional_params}

    # Add common default values
    context_params.setdefault("currency", settings.default_budget_currency)
    context_params.setdefault("user_location", settings.default_country)

    logger.info(
        "Generating context-aware prompt",
        prompt_type=prompt_type,
        language=language,
        context_params_count=len(context_params),
    )

    return render_system_prompt(prompt_type, language, **context_params)
