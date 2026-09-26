function feedback = generateRecaptureFeedback(metrics)
%GENERATERECAPTUREFEEDBACK Generate capture guidance from native quality heuristics.

    if metrics.iqs >= 0.7
        feedback = "Good quality image.";
        return;
    end
    messages = strings(0,1);
    if metrics.focus < 0.5
        messages(end+1) = "Image is out of focus. Please refocus and ensure patient is still.";
    end
    if metrics.illumination < 0.6
        messages(end+1) = "Uneven illumination detected. Check flash intensity and alignment.";
    end
    if metrics.fov < 0.5
        messages(end+1) = "Incomplete field of view. Ensure proper alignment with pupil.";
    end
    if isempty(messages)
        feedback = "Borderline image quality; review before screening.";
    else
        feedback = strjoin(messages, " ");
    end
end
