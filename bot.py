import logging
import numpy as np
import joblib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes
)

# --- CONFIGURATION ---
# 1. REVOKE the token you just posted in our chat.
# 2. Get the NEW token from BotFather.
# 3. Paste your NEW, SECRET token here.
# --- CONFIGURATION ---
# Get your token from BotFather on Telegram
BOT_TOKEN = "8408765719:AAG_THE_NEW_SECRET_PART_HERE" 
MODEL_PATH = "infant_screening_model.joblib"

# Define states for the conversation
(ASKING_QUESTION, DONE) = range(2)

# --- LOAD YOUR MODEL ---
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please make sure the model file is in the same directory as bot.py")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have scikit-learn and xgboost installed (`pip install scikit-learn xgboost`)")
    exit()


# --- DEFINE QUESTIONS AND LABELS ---
# These MUST be in the exact same order as your training data features
QUESTIONS = [
    "Does the infant make eye contact when talked to?",
    "Does the infant smile back when you smile?",
    "Does the infant respond to their name?",
    "Does the infant coo or babble?",
    "Does the infant turn toward your voice?",
    "Does the infant show interest in faces?",
    "Does the infant follow moving objects with their eyes?",
    "Does the infant reach to pick up objects?",
    "Does the infant try to imitate your facial expressions?",
    "Does the infant grasp toys?",
    "Does the infant seem to have normal muscle tone (not too stiff or floppy)?",
    "Has the infant lost any skills they once had?"
]

# This mapping comes from your notebook:
# LabelEncoder sorts alphabetically: 'High Risk' (0), 'Low Risk' (1), 'Moderate Risk' (2)
RISK_CATEGORIES = {
    0: "High Risk 🚩",
    1: "Low Risk ✅",
    2: "Moderate Risk ⚠️"
}

# --- BOT FUNCTIONS ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the screening conversation."""
    # Initialize the state for this user
    context.user_data['question_index'] = 0
    context.user_data['answers'] = []

    await update.message.reply_text(
        "Welcome to the Infant Developmental Screening Tool. 👶\n\n"
        "I will ask 12 'Yes' or 'No' questions. This is a screening tool, "
        "not a medical diagnosis.\n\n"
        "You can type /cancel at any time to stop."
    )
    # Ask the first question
    return await ask_question(update, context)


async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the current question with Yes/No buttons."""
    index = context.user_data.get('question_index', 0)

    if index < len(QUESTIONS):
        question_text = f"Question {index + 1} of {len(QUESTIONS)}:\n\n{QUESTIONS[index]}"

        # Create Yes/No buttons
        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data='1'),
                InlineKeyboardButton("No", callback_data='0'),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # If this is the first question (from /start), send a new message
        # Otherwise, edit the previous question message
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=question_text,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                text=question_text,
                reply_markup=reply_markup
            )

        return ASKING_QUESTION  # Stay in the "asking" state
    else:
        # All questions are answered, move to prediction
        return await predict_and_reply(update, context)


async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the answer (1 for Yes, 0 for No) and asks the next question."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button press

    # Save the answer (1 or 0)
    answer = int(query.data)
    context.user_data['answers'].append(answer)

    # Increment question index
    context.user_data['question_index'] += 1

    # Ask the next question
    return await ask_question(update, context)


async def predict_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Collects all answers, makes a prediction, and sends the result."""
    answers = context.user_data['answers']
    
    # Reshape the answers into the format the model expects (1 row, 12 columns)
    user_input = np.array(answers).reshape(1, -1)

    # Make prediction
    predicted_encoded = model.predict(user_input)
    
    # Map the numeric prediction (0, 1, or 2) back to the text label
    result_category = RISK_CATEGORIES.get(predicted_encoded[0], "Unknown")

    # Send the final result
    result_text = (
        f"--- Screening Complete ---\n\n"
        f"Predicted Result: **{result_category}**\n\n"
        "**Disclaimer:** This is not a medical diagnosis. It is a screening tool "
        "based on a predictive model. Please consult a healthcare professional "
        "for any concerns."
    )

    # Edit the last message (which was a question)
    if update.callback_query:
        await update.callback_query.edit_message_text(text=result_text, parse_mode="Markdown")
    
    # Clean up user data to be ready for the next /start
    context.user_data.clear()
    
    # End the conversation
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "Screening cancelled. Type /start to begin again."
    )
    context.user_data.clear()
    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    
    application = Application.builder().token(BOT_TOKEN).build()

    # Create the ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASKING_QUESTION: [
                # This handles the "Yes" (1) or "No" (0) button presses
                CallbackQueryHandler(handle_answer)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    
    print("Bot is running... Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()
