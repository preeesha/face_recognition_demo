export function calculatePersonTimes(metadata) {
  const { frame_count, actual_duration, person_detection_counts } = metadata;

  const timePerFrame = actual_duration / frame_count;

  const results = Object.entries(person_detection_counts).map(([name, frames]) => {
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

  return results;
}



// // Example usage:
// const metadata = {
//   frame_count: 44,
//   actual_duration: 30.21,
//   person_detection_counts: {
//     swati: 44,
//     prisha: 44,
//     nishant: 34,
//   },
// };

// console.log(JSON.stringify(calculatePersonTimes(metadata), null, 2));
