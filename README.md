# QullyChat

QullyChat app strives to offer best AI chat experience as possible, offering as many options to the user as possible and both LLM and Diffusion responses in the same chat.
The idea was born out of my frustration with some things not available in LMStudio and with some ways LMStudio does things.

## Road-to-v1 TO-DO

- [x] Implement chat window
    - [x] System prompt changing in chat
    - [x] LLM changing in chat
    - [x] Choosing settings profile next to the LLM in topbar
    - [x] Chat-bound settings edit
    - [x] Settings importance layers (Chat > Model > Profile)
    - [ ] Show generate button when sent with no model selected & prompt to choose model
    - [ ] Move responses to separate boxes (User R-aligned, Assistant L-alligned, System C-alligned)
    - [ ] Add copy button
    - [ ] Show statistics
    - [ ] Add regenerate button
    - [ ] Add branch chat function
    - [ ] Add delete button
    - [ ] Edit made prompts (but only with regenerate & branch solutions)
- [x] Implement chats list
    - [ ] Ability to move chats on the list
    - [ ] Export chat
    - [x] Renaming chats
- [x] Implement LLM model list
    - [x] Model-bound settings edit
    - [ ] Make columns correct wideness
    - [ ] Add more info
    - [ ] Models sorting
    - [ ] Models grouping (by same family, by same model and different weights/parameters)
    - [ ] Models filtering
- [ ] Generate requirements.txt
- [x] Add settings tab
    - [x] Settings profiles
    - [x] Default settings
    - [ ] System prompt profiles
    - [ ] Settings for statistics to show (incl. model used)
    - [ ] Settings for chat colors (not just color schemes)
    - [ ] Adjust column wideness
    - [ ] Add more settings
- [ ] Test tools options
    - [ ] Setting up many models with same prompt
    - [ ] Rate each step of responses
    - [ ] Compare statistics (benchmark)
- [ ] Store statistics in chat file
- [x] Starting llama.cpp server in app
- [x] Try to preserve system prompt history
- [ ] Implement Image Generation in chat
- [ ] Implement GenAI model list
- [ ] Adjust settings tab for GenAI models
- [ ] Package first release for github
- [ ] Make looks adjustments
    - [ ] ???

## v1-ready TO-DO

- [ ] Add installation instructions
- [ ] Document usage examples
- [ ] Implement model evaluation module
- [ ] Add contributing guidelines
- [ ] Feature overview

## After-v1 TO-DO

- [ ] Add Huggingface support