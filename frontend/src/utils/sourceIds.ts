export function sourceElementId(messageId: string, sourceId: string) {
  const safeMessageId = messageId.replace(/[^a-zA-Z0-9_-]/g, "-");
  const safeSourceId = sourceId.replace(/[^a-zA-Z0-9_-]/g, "-");
  return `source-${safeMessageId}-${safeSourceId}`;
}
