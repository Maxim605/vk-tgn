/**
 * Проверка покрытия данных перед экспортом.
 * Запуск: node learner/data/check_data.js
 */
const { Database } = require("arangojs");

const db = new Database({
  url: "http://localhost:8529",
  databaseName: "_system",
  auth: { username: "root", password: "test" },
});

async function main() {
  const checks = await Promise.all([
    // Всего событий write/comment
    db.query(`RETURN LENGTH(FOR e IN interactions FILTER e.type IN ['write','comment'] RETURN 1)`).then(c => c.next()),
    // Из них с created_at
    db.query(`RETURN LENGTH(FOR e IN interactions FILTER e.type IN ['write','comment'] AND e.created_at != null RETURN 1)`).then(c => c.next()),
    // write событий
    db.query(`RETURN LENGTH(FOR e IN interactions FILTER e.type == 'write' RETURN 1)`).then(c => c.next()),
    // comment событий
    db.query(`RETURN LENGTH(FOR e IN interactions FILTER e.type == 'comment' RETURN 1)`).then(c => c.next()),
    // Постов с непустым текстом
    db.query(`RETURN LENGTH(FOR p IN posts FILTER p.text != null AND p.text != '' RETURN 1)`).then(c => c.next()),
    // Комментариев с непустым текстом
    db.query(`RETURN LENGTH(FOR c IN comment FILTER c.text != null AND c.text != '' RETURN 1)`).then(c => c.next()),
    // Уникальных src в событиях
    db.query(`RETURN LENGTH(FOR e IN interactions FILTER e.type IN ['write','comment'] RETURN DISTINCT e._from)`).then(c => c.next()),
    // Уникальных dst (постов) в событиях
    db.query(`RETURN LENGTH(FOR e IN interactions FILTER e.type IN ['write','comment'] RETURN DISTINCT e._to)`).then(c => c.next()),
    // Временной диапазон событий
    db.query(`
      LET events = (FOR e IN interactions FILTER e.type IN ['write','comment'] AND e.created_at != null RETURN e.created_at)
      RETURN { min: MIN(events), max: MAX(events) }
    `).then(c => c.next()),
    // Сколько src из interactions НЕ в users (внешние)
    db.query(`
      LET event_srcs = (FOR e IN interactions FILTER e.type IN ['write','comment'] RETURN DISTINCT SPLIT(e._from,'/')[1])
      LET user_keys  = (FOR u IN users RETURN u._key)
      RETURN LENGTH(event_srcs[* FILTER CURRENT NOT IN user_keys])
    `).then(c => c.next()),
  ]);

  const [
    total_events, events_with_ts, write_count, comment_count,
    posts_with_text, comments_with_text,
    unique_src, unique_dst,
    time_range,
    external_users,
  ] = checks;

  console.log("=== Data Coverage Report ===\n");
  console.log(`Events (write+comment):     ${total_events}`);
  console.log(`  with created_at:          ${events_with_ts} (${(events_with_ts/total_events*100).toFixed(1)}%)`);
  console.log(`  write:                    ${write_count}`);
  console.log(`  comment:                  ${comment_count}`);
  console.log(`\nPosts with text:            ${posts_with_text}`);
  console.log(`Comments with text:         ${comments_with_text}`);
  console.log(`\nUnique src (users):         ${unique_src}`);
  console.log(`Unique dst (posts):         ${unique_dst}`);
  console.log(`External users (not in DB): ${external_users}`);
  console.log(`\nTime range:`);
  console.log(`  min: ${time_range.min}`);
  console.log(`  max: ${time_range.max}`);
}

main().catch(console.error);
