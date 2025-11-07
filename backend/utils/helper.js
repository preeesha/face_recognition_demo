export function calculatePersonTimes(metadata) {
  const { frame_count, actual_duration, person_detection_counts } = metadata;

  if (!frame_count || !actual_duration || !person_detection_counts) return [];

  const timePerFrame = actual_duration / frame_count;

  return Object.entries(person_detection_counts).map(([name, frames]) => {
    const timeSeconds = parseFloat((frames * timePerFrame).toFixed(2));
    const minutes = Math.floor(timeSeconds / 60);
    const seconds = Math.round(timeSeconds % 60);
    const timeFormatted = `${minutes}:${seconds.toString().padStart(2, "0")}`;

    return {
      name,
      frames_detected: frames,
      time_seconds: timeSeconds,
      time_formatted: timeFormatted,
    };
  });
}
